import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_unified import GPT_calls
import json
import time
import random

parallel_runs_amount = 10 # set 1 for sequential runs
stream = False # select True for streaming mode
combinations_to_test = 30

# List of models to test
model_names = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-0613" ]

# model_names = ["gpt-4-1106-preview"]

problem_1 = "A circus car rink has 12 red cars. They have 2 fewer green cars than they have red cars. They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars. If the rink has 75 cars in total how many yellow cars do they have?"
problem_2 = "Avi is trying to decide whether he really needs to do his homework. If the normal teacher comes in, there's a 40% chance she'll give everyone an extension. There's a 50% chance that tomorrow he'll have a substitute teacher who won't collect the homework. Even if the whole class doesn't get an extension, there's a 20% chance Avi can convince the teacher his dog ate his assignment and get a personal extension. What is the percentage chance that Avi will actually have to turn in his homework tomorrow?"


problems = [problem_1, problem_2]

for problem in problems:
    print(f"Testing problem: {problem}")
    system_message = "You are a perfect logician."

    # Split the problem into its individual sentences
    problem_sentences = problem.split(". ")

    # Generate all possible sentence combinations
    sentence_combinations = list(itertools.permutations(problem_sentences))
    print(f"Number of sentence combinations: {len(sentence_combinations)}")
    random.shuffle(sentence_combinations)
    # Function to make a call to claude and check if "24%" is in the response
    def check_claude_response(sentence_combination, model_name="gpt-4-turbo-2024-04-09"):
        GPT = GPT_calls(stream=stream, model=model_name)
        if system_message != "":
            GPT.add_message("system", system_message)
        reordered_problem = ". ".join(sentence_combination)
        GPT.add_message("user", reordered_problem)
        
        while True:
            try:
                response = GPT.get_response()
                break
            except Exception as e:
                # print(e)
                if "rate_limit_error" in str(e):
                    print("Rate limit error. Waiting for 30 seconds")
                    time.sleep(30)
                    continue
        if problem == problem_1:
            return "23" in response    
        else:
            return "24%" in response

    # Use ThreadPoolExecutor to make parallel calls
    for model_name in model_names:
        print(f"Testing model: {model_name}")
        results = {"system_message": system_message, "correct": 0, "incorrect": 0}
        with ThreadPoolExecutor(max_workers=parallel_runs_amount) as executor:
            future_to_combination = {executor.submit(check_claude_response, combination, model_name): combination for combination in sentence_combinations[:combinations_to_test]}
            for future in as_completed(future_to_combination):
                if future.result():
                    print("Correct")
                    results["correct"] += 1
                else:
                    print("Incorrect")
                    results["incorrect"] += 1

        # check how many files in the folder start with "results"
        import os
        files = os.listdir()
        results_files = [file for file in files if file.startswith("results")]

        # Write the results to a JSON file enumerating the file number
        with open(f'new_results__for_{model_name}_{len(results_files)}_for_{problem[:10]}.json', 'w') as f:
            json.dump(results, f)

        print("Results written to results.json")