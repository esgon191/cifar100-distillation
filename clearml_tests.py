from clearml import Task, Logger
from datetime import datetime
import argparse

# Тестирование работы clearml с парсингом аргументов
# (проверка подключения локально и с воркера в Colab)
parser = argparse.ArgumentParser(description="Пример: получение текста из аргументов командной строки")

parser.add_argument(
    "--test_text",
    type=str,
    required=True,
    help="Текст для теста (пример: --test_text 'Привет, мир!')"
)

args = parser.parse_args()

test_text = args.test_text

task = None
try:
    task = Task.init(
        project_name="cifar100-distillation",
        task_name="Тестирование подключение к ClearML из git-repo",
        task_type=Task.TaskTypes.testing,
        output_uri=False
    )
    Logger.current_logger().report_text(f"Test task initialized successfully {datetime.now()}")
    Logger.current_logger().report_text(f"Сообщение из аргументов: {test_text}")
    print("Успешно подключено к ClearML")

except Exception as e:
    print(e)

finally:
    if task is not None:
        task.close()
    print("Успешное завершение задачи")
