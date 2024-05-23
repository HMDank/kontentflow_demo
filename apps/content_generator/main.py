import gradio as gr
from openai import OpenAI
import re
import pandas as pd
import threading

import google.generativeai as genai

genai.configure(api_key="AIzaSyDPLqoVqSCBCLHluk06rwM18VObBfMjU78")

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config)

global PREV_LACKED_KEYWORDS, PREV_CONTENT_CACHE
max_iterations = 1
PREV_CONTENT_CACHE = {}
PREV_UNDERUSED_KEYWORDS = {}


# # Function to calculate keyword density and counts
def evaluate(text: str, keywords: list):
    '''
    Return the results of a text according to some fixed metrics
    :param text: The text body.
    :param keywords: A list of the keywords.
    :return: a tuple consisting of the original texxt, keyword coverage (%), underused keywords, missing keywords, and a dataframe showing the detailed results.
    '''
    temp_text = text.lower()
    words = re.findall(r"\w+", temp_text)

    results = []
    for keyword in keywords:
        temp_keyword = keyword.lower()
        keyword_length = len(temp_keyword.split())

        match_count = 0
        for i in range(len(words) - keyword_length + 1):
            segment = " ".join(words[i : i + keyword_length])
            if segment == temp_keyword:
                match_count += 1

        density = 100 * match_count / len(words) if words else 0
        results.append({"keyword": keyword,
                        "count": match_count,
                        "density %": round(density, 3)})

    results_df = pd.DataFrame(results)

    coverage = sum(result["count"] > 0
                   for result in results) / len(keywords)

    missing_keywords = ", ".join(results_df[results_df["density %"] == 0]["keyword"].tolist())

    return (text, coverage, missing_keywords, results_df)


def function_with_timeout(func, timeout_seconds=90, default_value=None):
    """
    Decorator to run a function with a timeout.
    :param func: The function to run.
    :param timeout_seconds: Maximum number of seconds to allow the function to run.
    :param default_value: The default value to return if the timeout is exceeded.
    :return: The function's result or the default value if the timeout is exceeded.
    """
    def wrapper(*args, **kwargs):
        result = [default_value]  # List to store the result

        def run_func():
            result[0] = func(*args, **kwargs)
        thread = threading.Thread(target=run_func)
        thread.start()
        thread.join(timeout_seconds)  # Wait for the specified timeout duration
        if thread.is_alive():
            # If the thread is still alive, it means the function did not complete in time
            raise TimeoutError
        else:
            return result[0]
    return wrapper


def generate_gemini_content(prompt):
    response = model.generate_content(prompt)
    return response.text


# Function to generate content using GPT-4
# def generate_gpt_content(prompt):
#     # Generate initial content
#     completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"{prompt}",
#             }
#         ],
#         model="gpt-4-1106-preview",
#     )

#     content = completion.choices[0].message.content
#     return content


# def gemini_test(keyword_string: str, limit: int):
#     print(f'counter = {limit}')
#     global PREV_OVERUSED_KEYWORDS, PREV_UNDERUSED_KEYWORDS, PREV_CONTENT_CACHE
#     keyword_list = [keyword.strip() for keyword in keyword_string.split(",")]

#     prev_content_cache = PREV_CONTENT_CACHE.get(keyword_string, "")
#     prev_underused_keywords = PREV_UNDERUSED_KEYWORDS.get(keyword_string, "")

#     if not prev_content_cache:
#         initial_prompt = f"""
#         You are a content marketer, write an SEO optimized content using Markdown format, using the following list of keywords.

#         The list of keywords:
#         {keyword_string}

#         The content should be written in a reader-friendly and approachable style. The ultimate goal is to help the article rank high in SEO rankings.

#         Please recognize the language of the keywords and use it to build the blog. It will consist of the following parts:

#         - Appealing headline: Capture the reader's attention and reflect the content of the blog.
#         - Introduction: Provide context, purpose, and main content of the blog.
#         - Body
#         - Conclusion: Summarize the main points of the blog and leave a lasting impression on the reader.

#         Please ensure the blog is fully written with a minimum length of 1000 words, and contains all of the keywords. If no keyword is omitted, I will tip you $100.
#         """
#     else:
#         initial_prompt = f"""
#         You are a content marketer specialized in writing SEO content. Rewrite the following text so that all of the keywords in the given keyword list are included in the final result.
#         Make sure that the final blog post in in Markdown format, contains at least 5 headings and all of the keywords listed below.
#         Keywords to add:
#         {prev_underused_keywords}

#         The Text:
#         {prev_content_cache}
#         """
#     generated_content = generate_gemini_content(initial_prompt)
#     results = evaluate(generated_content, keyword_list)

#     PREV_CONTENT_CACHE[keyword_string] = generated_content
#     PREV_UNDERUSED_KEYWORDS[keyword_string] = results[2]
#     good_condition = results[1] >= 0.95  # and underused_keywords.count(',') < 5
#     print(f"{results[1]}, {prev_underused_keywords}")
#     return results if good_condition or limit == 1 else gemini_test(keyword_string, limit - 1)


# demo = gr.Interface(
#     fn=gemini_test,
#     inputs=[gr.Textbox(label="Input Keywords (ngăn cách bằng dấu phẩy)"),
#             gr.Number(label="Limit")],
#     outputs=[
#         gr.Textbox(label="Bài viết"),
#         gr.Label(label="Độ phủ từ khoá"),
#         gr.Textbox(label="Các từ khoá cần thêm"),
#         gr.Dataframe(label="Thống kê từ khoá"),
#     ],
# )

# demo.launch(share=True, max_threads=1)