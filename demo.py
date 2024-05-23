from bs4 import BeautifulSoup
import io
import PIL
import gradio as gr
import threading
import markdown2

from apps.blog_meta_crawler.crawler import get_google_image_url
from apps.content_generator.main import gemini_test
from apps.watermark.main import apply_watermark

import cloudinary
from cloudinary.uploader import upload
cloudinary.config(
    cloud_name="dnwqhi8ln",
    api_key="632997457723792",
    api_secret="st8Swvubuf-u8QizaEfbc-8i7RY"
)

interval_limit = 5


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


def create_post(keywords: str,
                logo=1,
                position='top-left'):
    result = gemini_test(keywords, interval_limit)
    print('Finished Generating Text')
    main_text = result[0]
    html_text = markdown2.markdown(main_text)
    soup = BeautifulSoup(html_text, 'html.parser')
    theme = keywords.split(',')[0].strip()
    headings = soup.find_all(['h1', 'h2', 'h3'])
    url_list = function_with_timeout(get_google_image_url)(theme)
    if len(url_list) > 0:
        for index, heading in enumerate(headings):
            try:
                image = function_with_timeout(apply_watermark)(url_list[index], logo, 1, 'top-left').convert('RGB')
                if image is None:
                    continue
                image_data = io.BytesIO()
                image.save(image_data, format='JPEG', quality=70)
                image_data.seek(0)
                uploaded_image = upload(file=image_data)
                if uploaded_image is not None:
                    new_img_tag = soup.new_tag('img', src=uploaded_image["url"], alt=heading, style='width: 60%;')
                    heading.insert_after(new_img_tag)
            except (PIL.UnidentifiedImageError, OSError, TimeoutError):
                print("Unidentified Image Error")
            except Exception as e:
                print(f"Other Exception occurred: {e}")
    final = soup.prettify()
    return final, result[1], result[2], result[3]


demo = gr.Interface(
        fn=create_post,
        inputs=[gr.Textbox(label="Input Keywords (Input the most important keyword at the top; seperate the keywords by commas)"),
                gr.Textbox(label="Logo"),],
        outputs=[
            gr.Textbox(label="Bài viết"),
            gr.Label(label="Độ phủ từ khoá"),
            gr.Textbox(label="Các từ khoá cần thêm"),
            gr.Dataframe(label="Thống kê từ khoá"),
        ],
    )

demo.launch(share=True, max_threads=1)


                # gr.Textbox(label="Language"),
                # gr.Textbox(label="Logo (Input 1 or 2 for Vucar\'s blue and white logo respectively, or alternatively a Logo\'s URL)"),
                # gr.Textbox(label='Position (Input "top-left", "top-right", "bottom-left", "bottom-right", or "middle")')