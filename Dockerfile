FROM python:3.11

#workdir inside the container
WORKDIR /app

#copy pyproject first
COPY pyproject.toml .

#copy source code
COPY src/ ./src/

#install project and dependencies
#everything in pyproject.toml will be installed
#".[dev]" => so we can run tests via the docker image
RUN pip install --no-cache-dir ".[dev]"

#copying all project files into container
#everything from the main directory will be imported
COPY . .

CMD ["python", "-m", "src.main", "full_run"]