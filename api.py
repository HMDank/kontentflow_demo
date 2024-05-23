from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse

from typing import Annotated, Optional
from bs4 import BeautifulSoup
import uvicorn
import time
import io
import secrets
import threading
import PIL
import markdown2

from apps.blog_meta_crawler.crawler import (
    generate_vietnamese_sections_with_image_seo_keywords,
    generate_seo_data_with_images,
    get_google_image_url
)
from apps.content_generator.main import gemini_test
from apps.watermark.main import apply_watermark

import cloudinary
from cloudinary.uploader import upload

cloudinary.config(
    cloud_name="dnwqhi8ln",
    api_key="632997457723792",
    api_secret="st8Swvubuf-u8QizaEfbc-8i7RY"
)


app = FastAPI()

security = HTTPBasic()

interval_limit = 10

@app.get('/')
def home():
    return {'Use /docs'}


def authenticate_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> bool:
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"dankhoang"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"12345678"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


def function_with_timeout(func, timeout_seconds=20, default_value=None):
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

@app.get('/content-generator')
def generate_content(authenticated: bool = Depends(authenticate_user),
                     keywords: str = Query(...,
                                           description='Input keywords, \
                                           seperated by a comma (,)')):
    start = time.perf_counter()
    result = gemini_test(keywords, interval_limit)
    end = time.perf_counter()
    return {
        "time taken": f'{round(end - start, 2)}s',
        "generated_content": result[0],
        "coverage": result[1],
        "missing_keywords": result[2],
        "results_df": result[3].to_dict(),
    }


@app.get('/url-generator/images')
def initialize_blog_crawler(authenticated: bool = Depends(authenticate_user),
                            Text_body: str = Query(..., description='\
                                                   Input text here')):
    output = generate_vietnamese_sections_with_image_seo_keywords(Text_body)
    output_with_image_urls = generate_seo_data_with_images(output)
    return output_with_image_urls


@app.get('/watermark')
def lay_watermark(authenticated: bool = Depends(authenticate_user),
                  url: str = Query(..., description='Input image URL'),
                  logo: Optional[str] = Query('1', description='Input 1 or 2 for Vucar\'s blue and white logo respectively, or alternatively a Logo\'s URL'),
                  opacity: Optional[float] = Query(1, description='Input a number between 0 and 1'),
                  position: Optional[str] = Query('top-left', description='(Optional) Input "top-left",\
                      "top-right", "bottom-left", "bottom-right", or "middle"')):
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity should be between 0 and 1")
    if position not in ["top-left", "top-right",
                        "bottom-left", "bottom-right", "middle"]:
        raise ValueError('Position should be within the following: "top-left",\
                      "top-right", "bottom-left", "bottom-right", or "middle"')
    try:
        image = function_with_timeout(apply_watermark)(url, logo, opacity, position).convert('RGB')
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG', quality=70)
        image_data.seek(0)
        uploaded_image = upload(file=image_data)
    except (PIL.UnidentifiedImageError, OSError, TimeoutError) as e:
        print(f"Unidentified Image Error for URL at index {i}: {e}")
    except Exception as e:
        # Handle other exceptions if needed
        print(f"Other Exception occurred: {e}")
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_data.seek(0)
    uploaded_image = upload(file=image_data)
    return uploaded_image['url']


@app.get('/create-gpt-post')
def create_finished_post(keywords: str):

    result = gemini_test(keywords, interval_limit)
    generated_text = result[0]

    return markdown2.markdown(generated_text)

@app.get('/create-gemini-post')
def create_gemini_post(keywords: str = Query(..., description='Input keywords,\
                                           seperated by a comma (,)'),
                       logo: Optional[str] = Query('1', description='Input 1 or 2 for Vucar\'s blue and white logo respectively, or alternatively a Logo\'s URL'),
                       opacity: Optional[float] = Query(1, description='Input a number between 0 and 1'),
                       position: Optional[str] = Query('top-left', description='Input "top-left",\
                      "top-right", "bottom-left", "bottom-right", or "middle"')):
    result = gemini_test(keywords.lower(), interval_limit)
    print('Finished Generating Text')
    main_text = result[0]
    html_text = markdown2.markdown(main_text)
    soup = BeautifulSoup(html_text, 'html.parser')
    theme = keywords.split(',')[0].strip()
    print(theme)
    headings = soup.find_all(['h1', 'h2', 'h3'])
    url_list = function_with_timeout(get_google_image_url)(theme, len(headings))
    if len(url_list) > 0:
        for index, heading in enumerate(headings):
            try:
                uploaded_image = None
                image = function_with_timeout(apply_watermark)(url_list[index], logo, opacity, position).convert('RGB')
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
            except Exception:
                # Handle other exceptions if needed
                print("Other Exception occurred")

    final = soup.prettify()
    return HTMLResponse(content=final)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)