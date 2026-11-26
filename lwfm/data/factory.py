from lwfm.data.fakeclassification import FakeClassificationDataModule


def build_data_module(config, seed):
    """Convert the YAML data config into an instance of a DataModule"""
    data_registry = {
        "fake_classification": FakeClassificationDataModule,
    }

    data_type = config["type"]

    if data_type not in data_registry:
        return ValueError(f"{data_type} is not a defined dataset")

    return data_registry[data_type](
        seed,
        config["num_classes"],
        config["num_train"],
        config["num_val"],
        config["image_size"],
        config["batch_size"],
        config["num_workers"],
        **config.get("kwargs", {}),
    )
