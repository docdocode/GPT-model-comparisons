import os
import json
from openai import OpenAI, AsyncOpenAI
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import re
from exa_py import Exa
from pathlib import Path


class GPT_calls:
    def __init__(self, 
                 name="GPT chat",
                 api_key=None, model="gpt-4-turbo-preview", 
                 max_history_words=10000, 
                 max_words_per_message=None,
                 json_mode=False, stream=True, 
                 use_async=False,
                 exa_api_key=None,  # Add an argument for the Exa API key
                 perplexity_api_key=None  # Add an argument for the Perplexity API key
                 ):
        
        self.name = name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-"
        self.model = model
        self.history = []
        self.max_history_words = max_history_words
        self.max_words_per_message = max_words_per_message
        self.json_mode = json_mode
        self.stream = stream
        self.use_async = use_async


        # print the initialization status
        print(colored(f"{self.name} initialized with json_mode={json_mode}, stream={stream}, use_async={use_async}, max_history_words={max_history_words}, max_words_per_message={max_words_per_message}", "red"))

        if use_async:
            self.client = AsyncOpenAI(api_key=self.api_key)
        if not use_async:
            self.client = OpenAI(api_key=self.api_key)


    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            self.history.append({"role": role, "content": str(content) + f" please use {self.max_words_per_message} words or less"})
        elif role == "user" and self.max_words_per_message is None:
            self.history.append({"role": role, "content": str(content)})
        else:
            self.history.append({"role": role, "content": str(content)})

    def print_history_length(self):
        history_length = [len(str(message["content"]).split()) for message in self.history]
        print("\n")
        print(f"current history length is {sum(history_length)} words")

    def clear_history(self):
        self.history.clear()

    def chat(self, question, **kwargs):
        self.add_message("user", question)
        return self.get_response(**kwargs)

    def trim_history(self):
        words_count = sum(len(str(message["content"]).split()) for message in self.history if message["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history[0]["content"].split())
            self.history.pop(1)  # Remove second message because first is always system message

    #############################
    # Async versions of the above
    #############################

    async def add_message_async(self, role, content):
        if role == "user" and self.max_words_per_message:
            self.history.append({"role": role, "content": str(content) + f" please use {self.max_words_per_message} words or less"})
        elif role == "user" and self.max_words_per_message is None:
            self.history.append({"role": role, "content": str(content)})
        else:
            self.history.append({"role": role, "content": str(content)})

    async def clear_history_async(self):
        self.history.clear()
    
    async def chat_async(self, question, **kwargs):
        await self.add_message_async("user", question)
        return await self.get_response_async(**kwargs)
    
    async def trim_history_async(self):
        words_count = sum(len(str(message["content"]).split()) for message in self.history if message["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history[0]["content"].split())
            self.history.pop(1)

    def get_response(self, color="green", should_print=True, **kwargs):
        if self.json_mode:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True if self.stream else False,
                response_format={"type": "json_object"},
                **kwargs
            )

            if self.stream:
                assistant_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content: 
                        text_chunk = chunk.choices[0].delta.content 
                        if should_print:
                            print(colored(text_chunk, color), end="", flush=True)
                        assistant_response += str(text_chunk)

                # convert the str to json
                assistant_response = json.loads(assistant_response)

            else:
                assistant_response = json.loads(response.choices[0].message.content)

        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True if self.stream else False,
                **kwargs
            )

            if self.stream:
                assistant_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content: 
                        text_chunk = chunk.choices[0].delta.content 
                        if should_print:
                            print(colored(text_chunk, "yellow"), end="", flush=True)
                        assistant_response += str(text_chunk)
                print("\n")
            else:
                assistant_response = response.choices[0].message.content       
        
        self.add_message("assistant", str(assistant_response))
        self.trim_history()
        return assistant_response
    
    async def get_response_async(self, color="yellow", should_print=True, **kwargs):
        if self.json_mode:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True if self.stream else False,
                response_format={"type": "json_object"},
                **kwargs
            )

            if self.stream:
                assistant_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content: 
                        text_chunk = chunk.choices[0].delta.content 
                        if should_print:
                            print(colored(text_chunk, color), end="", flush=True)
                        assistant_response += str(text_chunk)

                # convert the str to json
                assistant_response = json.loads(assistant_response)

            else:
                assistant_response = json.loads(response.choices[0].message.content)

        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True if self.stream else False,
                **kwargs
            )

            if self.stream:
                assistant_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content: 
                        text_chunk = chunk.choices[0].delta.content 
                        print(colored(text_chunk, color), end="", flush=True)
                        assistant_response += str(text_chunk)
                print("\n")
            else:
                assistant_response = response.choices[0].message.content       
        
        await self.add_message_async("assistant", str(assistant_response))
        await self.trim_history_async()
        return assistant_response
    