MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\n Query: [question] \n\nAnswer:'

CODE_PROMPT = 'You are a helpful programmer, please help me answer the following questions with the following relevant contexts. \
I hope you understand the task description and give a {language} code example. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\n Task description: [question] \n\nAnswer:'

def wrap_prompt(question, context, prompt_id=4, language='python') -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = CODE_PROMPT.replace('[question]', question).replace('[context]', context_str).replace('{language}', language)
    else:
        input_prompt = CODE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

