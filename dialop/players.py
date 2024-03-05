import copy
import pdb
import hf_olmo
import json
import openai
import os
import pathlib
from rich.prompt import IntPrompt, Prompt
from rich.markup import escape
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from envs import DialogueEnv
from utils import num_tokens
from dialop.openai_utils import openai_caller

try:
    with open(pathlib.Path(__file__).parent / ".api_key") as f:
        x = json.load(f)
        openai.organization = x["organization"]
        openai.api_key = x["api_key"]
    print("Loaded .api_key")
except:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("Warning: no OpenAI API key loaded.")

class OutOfContextError(Exception):
    pass

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self)
      self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        #self.stops = stops
        for stop_id in self.stops:
            if input_ids[0][-1].item() == stop_id:
                return True

class DryRunPlayer:

    def __init__(self, prompt, role, console, task="planning"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.calls = 0
        self.task = task

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.calls += 1
        if self.role == "agent" and self.calls == 5:
            if self.task == "planning":
                return f" [propose] [Saul's, Cookies Cream, Mad Seoul]"
            elif self.task == "mediation":
                return f" [propose] User 0: [1], User 1: [15]"
        elif self.role == "user" and self.calls == 6:
            return f" [reject]"
        return f" [message] {self.calls}"

class LLMPlayer:

    def __init__(self, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", optional=None):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model = "text-davinci-003"
        self.optional = optional
        self.removed_optional = False
        if self.role in ["user", "agent", "user0", "user1"]:
            stop_tokens = ["User", "Agent", "You", "\n"]
        elif self.role in ["player-1", "player-2"]:
            stop_tokens = ["Partner", "You", "\n"]
        else:
            raise NotImplementedError
        self.model_kwargs = dict(
            model=self.model,
            temperature=0.1,
            top_p=.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_tokens,
        )
        if model_kwargs is not None:
            self.model_kwargs.update(**model_kwargs)
        self.prefix = prefix
    #    self.model = "gpt-3.5-turbo"

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.console.rule(f"{self.role}'s turn")
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        #self.console.print(escape(self.prompt))
        remaining = 4096 - num_tokens(self.prompt)
        if remaining < 0 and self.optional:
            self._remove_optional_context()
            remaining = 4096 - num_tokens(self.prompt)
        # Still out of context after removing
        if remaining < 0:
            print("OUT OF CONTEXT! Remaining ", remaining)
            raise OutOfContextError()
        kwargs = dict(
            prompt=self.prompt,
            max_tokens=min(remaining, 128),
        )
        kwargs.update(**self.model_kwargs)
        import pdb; pdb.set_trace()
        response = openai.completions.create(**kwargs)
        self.console.print("Response: ",
                           escape(response["choices"][0]["text"].strip()))
        self.console.print("stop: ", response["choices"][0]["finish_reason"])
        if response["choices"][0]["finish_reason"] == "length":
            if not self.optional:
                raise OutOfContextError()
            self._remove_optional_context()
            response = openai.completions.create(**kwargs)
            self.console.print("Response: ",
                               escape(response["choices"][0]["text"].strip()))
            self.console.print("stop: ", response["choices"][0]["finish_reason"])
        self.console.print(response["usage"])
        return response["choices"][0]["text"].strip()

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True


class OpenSourceLLMPlayer:

    def __init__(self, model_type, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", gpu_id=0, optional=None):
        self.prompt = prompt
        self.role = role
        self.console = console
        #self.model = "mistralai/Mistral-7B-Instruct-v0.2"
        #self.model = "mistralai/Mistral-7B-Instruct-v0.1"
        #self.model = "allenai/OLMo-7B"
        self.model_type = model_type
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_type, torch_dtype=torch.float16).to(self.device)

        self.optional = optional
        self.removed_optional = False
        if self.role in ["user", "agent", "user0", "user1"]:
            stop_tokens = ["User", "Agent", "You", "\n"]
        elif self.role in ["player-1", "player-2"]:
            stop_tokens = ["Partner", "You", "\n"]
        else:
            raise NotImplementedError
        stop_words_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))[-1] for x in stop_tokens]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])

        self.model_kwargs = dict(
            temperature=0.1,
            top_p=.95,
            do_sample=True,
            stopping_criteria=self.stopping_criteria
            #frequency_penalty=0,
            #presence_penalty=0,
            #stop=stop_tokens,
        )
        if model_kwargs is not None:
            self.model_kwargs.update(**model_kwargs)
        self.prefix = prefix
        self.final_sys_message = {
            "role": "system",
            "content": "Send the next message based on the conversation so far. \n If you wish to communicate with your partner, begin your message with [message]. If you wish to propose an assignment, your message must start with [propose], followed by 'Proposal:' and have 8 lines, where each line is of the format Paper: Reviewer. If your partner just proposed an assignment, you must respond with [accept] or [reject]. You can only send [accept] or [reject] after receiving a [propose] message from your partner."
        }
    #    self.model = "gpt-3.5-turbo"

    def observe(self, obs):
        try:
            role, obs_string = obs
        except Exception as e:
            pdb.set_trace()
        #print(f"Player {self.role} adding observation: from {role}, {obs_string}")
        #pdb.set_trace()
        self.prompt.append({'role': role, 'content': obs_string})

    def stringify_prompt_message(self, messages):
        out_str = "" 
        for m in messages:
            if m['role'] == 'system':
                out_str += '\n' + m['content'] + '\n'
            elif m['role'] == 'user':
                out_str += 'Player: ' + m['content'] + '\n '
            elif m['role'] == 'assistant':
                out_str += 'You: ' + m['content'] + '\n '
        out_str += 'You: '
        return out_str

    def mistralify_messages(self, messages):
        new_input_messages = []
        for m in messages:
            if m['role'] == 'system':
                m['role'] = 'user'
                new_input_messages.append(m)
            else:
                if m['role'] == 'user':
                    if m['content'].startswith("NEW SET:"):
                        message_content = m['content'] + " \n"
                    else:
                        message_content = "Partner: " + m['content'] + " \n "
                elif m['role'] == 'assistant':
                    message_content = "You: " + m['content'] + " \n "
                if new_input_messages[-1]['role'] == 'assistant':
                    new_input_messages[-1]['content'] += message_content
                else:
                    new_input_messages.append({
                        'role': 'assistant',
                        'content': message_content
                    })
        return new_input_messages

    def respond(self, do_print=False):
        remaining = 4096# - num_tokens(self.prompt)
        kwargs = dict(
            max_new_tokens=min(remaining, 128),
        )
        kwargs.update(**self.model_kwargs)
        input_messages = self.prompt #+ [self.final_sys_message]
        #input_messages = self.mistralify_messages(input_messages)
        #pdb.set_trace()
        #encodings = self.tokenizer.apply_chat_template(input_messages, return_tensors="pt").to(self.device)

        input_str = self.stringify_prompt_message(input_messages)
        encodings = self.tokenizer(input_str, return_tensors='pt', return_token_type_ids=False).to(self.device)
        outputs = self.model.generate(**encodings, **kwargs)
        generated_ids = outputs[0, encodings.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = response.replace('\n', '<br/>')

        if do_print:
            self.console.print(f"{self.role.upper()} response: ",
                            response)

        #self.console.print(f"{self.role.upper()} response: ",
        #                   response.strip())
        #pdb.set_trace()
        #self.console.print("stop: ", response["choices"][0]["finish_reason"])
        #if response["choices"][0]["finish_reason"] == "length":
        #    if not self.optional:
        #        raise OutOfContextError()
        #    self._remove_optional_context()

        #    input_messages = [{'role': 'user', 'content': self.prompt}]
        #    input_ids_encoded = self.tokenizer.apply_chat_template(input_messages, return_tensors="pt").to(self.device)
        #    outputs = self.model.generate(input_ids_encoded, **kwargs)
        #    generated_ids = outputs[0, input_ids_encoded.shape[1]+1:]
        #    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        #    self.console.print("Response: ",
        #                       escape(response["choices"][0]["text"].strip()))
        #    self.console.print("stop: ", response["choices"][0]["finish_reason"])
        #self.console.print(response["usage"])
        pdb.set_trace()
        return response.strip()

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True


class ChatGPTPlayer:

    def __init__(self, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", gpu_id=0, optional=None):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.optional = optional
        self.removed_optional = False
        if self.role in ["user", "agent", "user0", "user1"]:
            stop_tokens = ["User", "Agent", "You", "\n"]
        elif self.role in ["player-1", "player-2"]:
            stop_tokens = ["Partner", "You", "\n"]
        else:
            raise NotImplementedError
        #stop_words_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))[-1] for x in stop_tokens]
        #self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])
        self.final_sys_message = {
            "role": "system",
            "content": "Send the next message based on the conversation so far. \n If you wish to communicate with your partner, begin your message with [message]. If you wish to propose an assignment, your message must start with [propose], followed by 'Proposal:' and have 8 lines, where each line is of the format Paper: Reviewer. If your partner just proposed an assignment, you must respond with [accept] or [reject]. You can only send [accept] or [reject] after receiving a [propose] message from your partner."
        }

        self.model_kwargs = dict(
            temperature=0.5,
            top_p=.95,
            frequency_penalty=0,
            presence_penalty=0,
            #stop=stop_tokens,
        )
        if model_kwargs is not None:
            self.model_kwargs.update(**model_kwargs)
        self.prefix = prefix
        #print(f"Loaded ChatGPT player: {self.role}")
    #    self.model = "gpt-3.5-turbo"

    def observe(self, obs):
        try:
            role, obs_string = obs
        except Exception as e:
            pdb.set_trace()
        #print(f"Player {self.role} adding observation: from {role}, {obs_string}")
        #pdb.set_trace()
        self.prompt.append({'role': role, 'content': obs_string})

    def respond(self, do_print=False):
        #self.console.rule(f"{self.role}'s turn")
        remaining = 4096# - num_tokens(self.prompt)
        kwargs = dict(
            max_new_tokens=min(remaining, 128),
        )
        kwargs.update(**self.model_kwargs)
        input_messages = self.prompt + [self.final_sys_message]
        #print(len(input_messages))
        response = openai_caller(input_messages, model='chatgpt', **kwargs)
        #pdb.set_trace()

        response = response.replace('\n', '<br/>')

        if do_print:
            self.console.print(f"{self.role.upper()} response: ",
                            response)
        return response.strip()

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True


class HumanPlayer:

    def __init__(self, prompt, role, console, prefix="\nYou:"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.prefix = prefix

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        self.console.rule(f"Your turn ({self.role})")
        self.console.print(escape(self.prompt))
        resp = ""
        if self.prefix.strip().endswith("You to"):
            id_ = Prompt.ask(
                escape(f"Choose a player to talk to"),
                choices=["0","1","all"])
            resp += f" {id_}:"
        mtypes = ["[message]", "[propose]", "[accept]", "[reject]"]
        choices = " ".join(
                [f"({i}): {type_}" for i, type_ in enumerate(mtypes)])
        type_ = IntPrompt.ask(
                escape(
                    f"Choose one of the following message types:"
                    f"\n{choices}"),
                choices=["0","1","2","3"])
        message_type = mtypes[type_]
        if message_type not in ("[accept]", "[reject]"):
            content = Prompt.ask(escape(f"{message_type}"))
        else:
            content = ""
        resp += f" {message_type} {content}"
        return resp
