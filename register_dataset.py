"""
Регистрация CIFAR-100 (pickle train/test/meta) как ClearML Dataset на files server
ClearML. Запускается один раз локально; дальше train.py берёт данные через
--dataset_name. Пример:
    python register_dataset.py --data_dir data/cifar_100
"""
import argparse
import os

# Dataset.create принудительно ставит задаче output_uri=True, т.е. default_output_uri
# из ~/clearml.conf (S3). Переопределяем его на files server до импорта clearml
os.environ.setdefault("CLEARML_DEFAULT_OUTPUT_URI", "https://files.clear.ml")

from clearml import Dataset
from clearml.backend_api.session import Session

from kd.data import CIFAR100_FILES, load_cifar100


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Загрузка CIFAR-100 в ClearML Dataset")
    p.add_argument("--data_dir", type=str, default="data/cifar_100", help="Папка с train/test/meta")
    p.add_argument("--dataset_name", type=str, default="cifar100", help="Название датасета в ClearML")
    p.add_argument("--dataset_project", type=str, default="cifar100-distillation", help="Проект в ClearML")
    p.add_argument("--dataset_version", type=str, default="1.0.0", help="Версия датасета")
    return p


def main():
    args = get_parser().parse_args()

    # Проверка, что файлы читаются и это действительно CIFAR-100
    data = load_cifar100(args.data_dir, download=False)
    assert data.x_train.shape == (50000, 3, 32, 32) and data.x_test.shape == (10000, 3, 32, 32)

    ds = Dataset.create(
        dataset_name=args.dataset_name,
        dataset_project=args.dataset_project,
        dataset_version=args.dataset_version,
        description="CIFAR-100, python-версия (pickle train/test/meta)",
    )
    for name in CIFAR100_FILES:
        ds.add_files(path=os.path.join(args.data_dir, name))

    # Адрес files server явно: default_output_uri в ~/clearml.conf указывает на S3
    ds.upload(output_url=Session.get_files_server_host(), show_progress=True)
    ds.finalize()
    print(f"Датасет {args.dataset_name} v{args.dataset_version} зарегистрирован, id={ds.id}")


if __name__ == "__main__":
    main()
