# Начало работы

Репозиторий представляет пример FastApi в Docker

Для запуска контейнера необходимо создать `.env` файл и указать следующие переменные:

```sh
PORT=8000 - Порт, на котором будет запущен FastApi
````

Для запуска контейнера запустите следующую команду:
```sh
bash ./tools/run_fastapi sh
```

Пример запросов на python если FastApi запущен по адресу http://localhost:8000/
```python
from httpx import AsyncClient
import requests

if __name__ == '__main__':
   response = requests.get("http://localhost:8007/get_predict")
```

