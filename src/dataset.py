def get_prompts(
    dataset : str
):
    if dataset.lower() == "factscore":
        with open('data/factscore_names.txt', 'r') as fp:
            names = fp.readlines()

        prompts = [
            f"Please write one biographical paragraph about {name}."
            for name in names
        ]
        return prompts
    else:
        raise ValueError("Unsupported data set.")