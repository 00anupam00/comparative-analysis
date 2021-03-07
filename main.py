from src import DataLoader, Pipeline


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = DataLoader.load_data()
    Pipeline.create_pipeline(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
