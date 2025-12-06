def sum_n(*args):
    return sum(args)


def main():
    numbers = [1, 2, 3, 4]
    print(sum_n(*numbers))


if __name__ == "__main__":
    main()


