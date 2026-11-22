from lwfm.data.fakeclassification import FakeClassificationDataModule

def build_data_module(config):
    data_registry = {
        "fake_classification": FakeClassificationDataModule,
    }


    data_type = config["type"]

    if data_type not in data_registry:
        return ValueError(f"{data_type} is not a defined dataset")

    return data_registry[data_type](
        **config.get("kwargs", {})
    )