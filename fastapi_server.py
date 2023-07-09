import os
import subprocess
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
import time
import numpy as np

from ner_server import Inferer as NerInferer
from er_server import Inferer as ErInferer

class Text(BaseModel):
    text: str


app = FastAPI()

path_ner = os.getenv("PATH_NER")
path_er = os.getenv("PATH_ER")

ner = NerInferer(task="ner", path=path_ner)
er = ErInferer(path_ner=path_ner, path_er=path_er)


with open("./s3_models.txt", "r") as file:
    readed_file = file.read()
    models_titles = readed_file.split("\n")
models_titles = [model.replace(".zip", "") for model in models_titles]

models: Dict[str, dict] = {
    "ner": {"model": None, "get_data": None, "launch_function": ner.infer},
    "relation_extraction": {
        "model": None,
        "get_data": None,
        "launch_function": er.infer,
    },
}


# TODO добавить функцию для тестирования моделей на новых данных 
@app.post("/infer/{app_name}")
def infer(app_name: str, text: Text):
    """[summary]
    requests.post(f"{url}/infer/ner", json={"text": "РАНХиГС"})
    requests.post(f"{url}/infer/clf", json={"text": "Мясные консервы"})
    requests.post(f"{url}/infer/relation_extraction",
    json={"text":
          "Деньги на приобретение топлива Киев получил от Всемирного банка"})
    requests.post(f"{url}/infer/clf_news", json={"text": "Новость про что-то"})

    Args:
        app_name (str): [description]
        text (Text): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    preds = None
    if app_name == "ner":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "relation_extraction":
        preds = models[app_name]["launch_function"](text.text)
    else: 
        raise Exception("not supported")

    return preds

@app.get("/models_names")
def get_models_names():
    """
    Возвращает словарь с названиями моделей из файла s3_models.
    """
    return {
        "ner": models_titles[0],
        "relation_extraction": models_titles[1],
    }
