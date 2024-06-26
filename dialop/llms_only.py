import pdb
import json
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from rich import print
from rich.console import Console
import tyro
from dialop.envs import (
        OptimizationEnv, PlanningEnv, MediationEnv
        )
from dialop.players import HumanPlayer, LLMPlayer, OpenSourceLLMPlayer, ChatGPTPlayer
from dialop.utils import run, run_multiagent
from dialop.openai_utils import openai_caller
from dialop.games.optimization import WORKERS, TASKS

def load_prompt(game, player):
        fname = f"{game}_{player}.txt" if game != "optimization" else f"{game}.json"
        return json.load(open(Path(__file__).parent / f"prompts/{fname}"))

class CustomConsole:
    def __init__(self, console: Console, do_print=False):
        self.console = console
        self.do_print = do_print
    
    def rule(self, *inp):
        if self.do_print:
            self.console.rule(*inp)
    
    def print(self, *inp):
        if self.do_print:
            self.console.print(*inp)

def convert_assignment(assignment):
  converted = []
  #pdb.set_trace()
  for a in assignment:
    converted.append(f"{WORKERS[a[0]]}: {TASKS[a[1]]}")
  return converted

def main(
    game: Literal["optimization", "planning", "mediation"],
    max_length: int = 30,
    num_rollouts: int = 1,
    do_print: bool = False,
    model_type: str = 'chatgpt'
    ):
    c = Console()
    console = CustomConsole(c, do_print=do_print)
    if game == "optimization":
        env = OptimizationEnv()
    elif game == "planning":
        env = PlanningEnv()
    else:
        env = MediationEnv()


    out_file = f"outputs/{model_type.replace('/', '_')}_{game}_{num_rollouts}rollouts.json"
    outputs = []
    pbar = tqdm(total=num_rollouts)
    total_score, total_done, mean_score, total_failures, num_attempts = 0.0, 0, 0, 0, 0
    while total_done < num_rollouts:
        num_attempts += 1
        if do_print:
            print(f"Attempt #{num_attempts}")

        env.reset()
        if model_type =='chatgpt':
            players = {
                p: ( ChatGPTPlayer(load_prompt(game, p), p, console, gpu_id=i))
                for i, p in enumerate(env.players)
            }
        else:
            players = {
                p: ( OpenSourceLLMPlayer(model_type, load_prompt(game, p), p, console, gpu_id=i))
                for i, p in enumerate(env.players)
            }

        env, obss, history, observations = run(console, env, players, max_length, do_print)
        if obss['done'] is False:
            total_failures += 1
            if do_print:
                print(f"Failed to find an assignment in time")
            pbar.set_description(f"{total_done} assignments made, mean score: {mean_score:.4f}, cost=${openai_caller.compute_cost():.2f}")
            continue

        output_dict = {
            'best_assignment': convert_assignment(env.game.best_assignment),
            'best_assignment_reward': env.game.best_assignment_reward,
            'cost': openai_caller.compute_cost(),
            'dialog_history': history,
            'player_observations': observations,
            'done': obss['done'],
            'proposed_assignment': convert_assignment(env.game.proposal) if obss['done'] is True else [], 
            'proposal_reward': env.game.proposal_reward if obss['done'] is True else 0,
            'score': obss['info']['score_norm'] if obss['done'] is True else 0
        }
        #pdb.set_trace()
        outputs.append(output_dict)
        total_score += obss['info']['score_norm'] if obss['done'] is True else 0
        total_done += 1 if obss['done'] is True else 0
        mean_score = total_score/total_done if total_done > 0 else 0.0
        pbar.set_description(f"{total_done} assignments made, mean score: {mean_score:.4f}, cost=${openai_caller.compute_cost():.2f}")
        pbar.update(1)
        try:
            json.dump(outputs, open(out_file, 'w'), indent=2)
        except Exception as e:
            pdb.set_trace()


#if __name__ == '__main__':
#    main()
tyro.cli(main)