def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "teacher: интеграционные тесты на настоящем учителе из ClearML (нужны сеть и данные)",
    )
