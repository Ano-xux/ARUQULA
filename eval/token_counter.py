
import pandas as pd
import numpy as np
import tiktoken as tk
from tqdm import tqdm
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


model_info = {
    "model" : "gpt-4.1-mini-2025-04-14",
    "input_price" : 0.4, # Price per mil tokens
    "ouput_price" : 1.6,
    "encoder": "o200k_base"
}

tokeniser_model = tk.get_encoding(model_info["encoder"])

with open("controller.prompt") as f:
    start_tokens = f.readlines()
    start_tokens = len(tokeniser_model.encode("".join(start_tokens)))




def lines_to_text(end_lines: list, section: int):
    """
    Args: endlines -> List of of all lines in a segment in which a given section ends,
          section -> which section between lines is choosen -> start with 1

    return -> converts the list of lines to text in a dataframe (Index = real line in document)
              if section is out of bounds of end_lines return None
    """
    r_lines = []
    if section >= len(end_lines):
        return
    with open("spinach.log", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            r_lines.append(line)
        df = pd.DataFrame(r_lines, columns=["content"]).reset_index()
        df['index'] = df['index'] + 1
        df = df.set_index('index')
        return df.loc[end_lines[section-1]: end_lines[section]]
    

def organise(question_number, step_number = 0, part = None):
    """
    For performant use, use get_step with this function(question_number) as arg
    step_number/ part only for debugging
    returns lines_to_text() in which the Whole Text/Question/Prompt can be categorised
    """
    end_lines = []
    i = 0
    with open("spinach.log", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            i = i+1
            if line.find("initializing") != -1:
                end_lines.append(i)
        question_text = lines_to_text(end_lines, question_number)
    if not isinstance(question_text, pd.DataFrame):
        return "not valid promt number"
    for i in range(4):
        question_text.drop(question_text.index[-1], inplace=True)
    if step_number == 0:
        return question_text
    elif step_number != 0:
        end_lines = []
        for line, content in question_text.iterrows():
            if content["content"].find("#  Q:") != -1:
                end_lines.append(line-2)
        question_text = lines_to_text(end_lines, step_number)
        if not isinstance(question_text, pd.DataFrame):
            return "not valid step number"
        question_text = filter_out(question_text)
        if part == None:
            return question_text
        else:
            end_lines = [question_text.iloc[0].name]
            for line, content in question_text.iterrows():
                if content["content"].find("Current-Action") != -1:
                    end_lines.append(line-4)
            end_lines.append(question_text.iloc[-1].name)
            return lines_to_text(end_lines, part)
        

        
def get_step(prompt: pd.DataFrame, step_n, part):
    """
    prompt:pd.dataframe -> Text of a question as a Dataframe
    part:int : current(1) or past action(2)
    returns

    """
    end_lines = []
    for line, content in prompt.iterrows():
        if content["content"].find("#  Q:") != -1:
            end_lines.append(line-2)
    question_text = lines_to_text(end_lines, step_n)
    if not isinstance(question_text, pd.DataFrame):
        return "not valid step number"
    question_text = filter_out(question_text)
    end_lines = [question_text.iloc[0].name]
    for line, content in question_text.iterrows():
        if content["content"].find("Current-Action") != -1:
            end_lines.append(line-4)
    end_lines.append(question_text.iloc[-1].name)
    return lines_to_text(end_lines, part)
        

def filter_out(df: pd.DataFrame):
    """
    Filters out filler of prompt for tokenn counter
    returns pd.Dataframe
    """
    error_line = 0
    for line, content in df.iterrows():
        if content["content"].startswith("#########################################"):
            df.loc[line] = ""
        if content["content"].startswith("Traceback") and error_line == 0:
            error_line = line
            break
    if error_line != 0:
        df = df.loc[:error_line - 3]
    return df
        

actions_dict = {
    "search_class" :"search_class_by_label",
    "search_entity": "search_entity_by_label",
    "search_property": "search_property_by_label",
    "get_knowledgegraph_entry": "get_knowledgegraph_entry",
    "get_property_examples": "get_property_examples",
    "execute_sparql" : "execute_sparql",
}

keys = list(actions_dict.keys())
action_list = []
for i in keys:
    action_list.append(actions_dict[i])


def get_tokens_of_action(prompt_n, action):
    searched_action = "Action: " + actions_dict[action]
    prompt = organise(prompt_n)
    right_actions = []
    n = 1
    for i in range(16):
        step = get_step(prompt, i+1, 2)
        if isinstance(step, pd.DataFrame):
            for line, content in step.iterrows():
                if content["content"].startswith(searched_action):
                    n = n+1
                    right_actions.append(step.to_string(index=False))
    return np.round((len(tokeniser_model.encode("".join(right_actions)))/n))


def get_writing_tokens(prompt_n):
    failed_tries = 0
    all_current_action = []
    prompt = organise(prompt_n)
    for i in tqdm(range(16)):
        if not isinstance(get_step(prompt, i+1, 2), pd.DataFrame):
            if failed_tries > 0:
                break
            else:
                failed_tries = failed_tries+1
                continue
        all_current_action.append(get_step(prompt, i+1, 2).to_string(index=False))
    return len(tokeniser_model.encode("".join(all_current_action)))


def get_reading_tokens(prompt_n):
    failed_tries = 0
    tokens = 0
    prompt = organise(prompt_n)
    for i in tqdm(range(16)):
        if not isinstance(get_step(prompt, i+1, 2), pd.DataFrame):
            if failed_tries > 0:
                break
            else:
                failed_tries = failed_tries+1
                continue
        prompt_string = "".join(get_step(prompt, i+1, 1).to_string(index=False))
        c_tokens =len(tokeniser_model.encode(prompt_string))
        tokens = 2*tokens+c_tokens #Tokens in the token_counter + tokens from all last steps + tokens from current step 
    return tokens


def print_total_tokens(prompt_n):
    writing_tokens = get_writing_tokens(prompt_n)
    reading_tokens = get_reading_tokens(prompt_n)
    print("writing-tokens: ", writing_tokens)
    print("writing price:", writing_tokens*model_info["ouput_price"]/1000000, "$")
    print("reading tokens: ", reading_tokens)
    print("reading price:", reading_tokens*model_info["input_price"]/1000000, "$")


def approximate_total_value(closeness):
    """
    Arg: closeness-> number of iterations of total tokens which are used for approximation of total cost of spinach_log
    """
    total_read_tokens = 0
    total_write_tokens = 0
    for i in range(closeness):
        total_write_tokens = total_write_tokens + get_writing_tokens(i+1)
        total_read_tokens = total_read_tokens + get_reading_tokens(i+1)


    closeness_modifier = 248/closeness
    total_read_price = total_read_tokens*(model_info["input_price"]/1000000)*closeness_modifier
    total_write_price = total_write_tokens*(model_info["ouput_price"]/1000000)*closeness_modifier
    print(total_read_price, "$ : Reading price")
    print(total_write_price,  "$ : Writing Price")
    print(total_read_price+ total_write_price, "# :total price approximation for 248 questiosn + iterations for approximation: ", closeness)

def approximate_tokens_per_action(closeness):
    action_token_dict = {
    "search_class" :0,
    "search_entity": 0,
    "search_property": 0,
    "get_knowledgegraph_entry": 0,
    "get_property_examples": 0, #wenig verwendet: bei meisten iteration = 0
    "execute_sparql" : 0,
    }
    for action in tqdm(keys):
        action_tokens = 0
        for j in range(closeness):
            action_tokens = action_tokens + get_tokens_of_action(j+1, action)
        action_token_dict[action] = action_tokens/closeness
    return action_token_dict


ax = plt.subplot()
values = approximate_tokens_per_action(2)
keys = list(values.keys())
action_list = []
for i in keys:
    action_list.append(values[i])

a_color = ["blue", "yellow", "brown","green", "red", "orange"]
placeholder = ["A", "B", "C", "D", "E", "F"]

plt.bar(placeholder, action_list, width= 0.6, color = a_color)
plt.xlabel("action")
plt.ylabel("number of tokens")
ax.set_axisbelow(True)
plt.title("Tokens per action executed")
plt.grid(axis='y', linestyle='solid', alpha=0.6, c= "gray")
plt.savefig("action_tokens.png")




        








    
    


