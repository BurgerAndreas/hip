class Registry:
    mapping = {"model_name_mapping": {}}

    @classmethod
    def register_model(cls, name):
        def wrap(model_cls):
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"][name]


registry = Registry()
