# Use an official Python runtime as a parent image
FROM python:3.11

# Set environment variables
ENV POETRY_VERSION=1.7.1 \
    PYTHONUNBUFFERED=1

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY . /code

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["poetry","run","uvicorn","apps.api:app","--reload"]
