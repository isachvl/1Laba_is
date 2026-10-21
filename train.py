from experiment import Experiment


def main():

    experiment = Experiment("configs/base.yaml")

    experiment.run()


if __name__ == "__main__":
    main()