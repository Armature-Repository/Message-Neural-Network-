import random

def convert_data(data):
    clean_data = []
    for sample, result in data:
        clean_sample = [
            0 if char == "A" else
            1 if char == "B" else
            2 if char == "C" else
            3 if char == "D" else -1
            for char in sample
        ]
        clean_result = {"ATTACK": 0, "DEFEND": 1, "RETREAT": 2, "HOLD": 3}[result]
        clean_data.append((clean_sample, clean_result))
    return clean_data


def generate_data(n=5000):
    data = []
    for _ in range(n):
        length = random.randint(3, 7)
        sample = [random.choice("ABCD") for _ in range(length)]

        a_count = sample.count("A")
        b_count = sample.count("B")
        c_count = sample.count("C")
        d_count = sample.count("D")

        # Rule priority:
        # 1. Starts with A → ATTACK
        # 2. Ends with D → DEFEND
        # 3. C is most common → RETREAT
        # 4. Otherwise → HOLD
        
        if sample[0] == "A":
            result = "ATTACK"
        elif sample[-1] == "D":
            result = "DEFEND"
        elif c_count > a_count and c_count > b_count and c_count > d_count:
            result = "RETREAT"
        else:
            result = "HOLD"

        data.append((sample, result))

    return convert_data(data)


def save_to_file(filename="data.txt", n=5000):
    clean_data = generate_data(n)
    with open(filename, "w") as f:
        for sample, label in clean_data:
            f.write(f"{sample},{label}\n")
    print(f"Saved {n} samples to {filename}")


if __name__ == "__main__":
    save_to_file("data.txt", n=5000)