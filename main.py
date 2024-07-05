from loader import Loader
from grid import Grid
from split_preprocess import SplitPreprocessData

def main():
    ld = (Loader()
        .load_via_disk()
        .profile()
        .get_df())
    print(ld.head())
    split_data = SplitPreprocessData(ld).split_preprocess()
    Grid(split_data).run().show_result().pick_best_result().run_test_data()

if __name__ == "__main__":
    main()