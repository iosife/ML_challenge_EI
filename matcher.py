""" Matches the entries of two datasets. """

import sys
import logging as lg
import re
import difflib
import random
import pandas as pd

lg.basicConfig(format='%(asctime)s - %(message)s', level=lg.INFO)

class Matcher:
    """
    Matches the entries of two datasets.
    """

    def __init__(self, dataset1_filename: str, dataset2_filename: str) -> None:
        """
        1st arg: Filename of the 1st dataset.
        2st arg: Filename of the 2nd dataset.
        Returns: None.
        """
        self.dataset1_filename = dataset1_filename
        self.dataset2_filename = dataset2_filename
        self.dataset1 = pd.DataFrame()
        self.dataset2 = pd.DataFrame()
        self.matches = {}
        self.full_matches = 0
        self.partial_matches = 0
        self.evaluation_f1 = 0.0

    @staticmethod
    def load_dataset(dat_fname: str) -> pd.DataFrame:
        """
        Loads the two datasets.
        1st arg: The filename of the dataset
        Returns: The dataset as pandas datframe.
        """
        try:
            tsv_dataset = pd.read_csv(dat_fname, sep='\t',\
                error_bad_lines=False, warn_bad_lines=False)
        except IOError:
            lg.critical("Fatal error: dataset cannot be loaded.")
            sys.exit("Fatal error: dataset cannot be loaded.")
        return tsv_dataset

    @staticmethod
    def estimate_match_score(str1: str, str2: str) -> float:
        """
        Estimates a score indicating the lexical similarity (match)
        of two strings.
        1st arg: The first string.
        2nd arg: The second string.
        Returns: A similarity score (high means high similarity).
        """
        str_similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        str_similarity = ("%.2f" % str_similarity)
        return str_similarity

    def preprocess_datasets(self) -> None:
        """
        Preprocessing for both dataset:
        (i) Removal of duplicate entries.
        (ii) Substitution of missing fields with fixed negative values.
        (iii) Definition of data types for dataframes.
        (iv) Conversion to lower case.
        Returns: None.
        """
        self.dataset1 = self.dataset1.drop_duplicates()
        self.dataset2 = self.dataset2.drop_duplicates()
        self.dataset1 = self.dataset1.fillna(-1)
        self.dataset2 = self.dataset2.fillna(-2)

        new_dtypes1 = {"id": int, "name": str, "street_number": int, "street_type": str,\
            "street_name": str, "address_line2": str, "postal_code": int, "city": str}
        self.dataset1 = self.dataset1.astype(new_dtypes1)
        new_dtypes2 = {"address": str, "website": str, "id": int, "name": str}
        self.dataset2 = self.dataset2.astype(new_dtypes2)

        self.dataset1['name'] = self.dataset1['name'].str.lower()
        self.dataset1['street_type'] = self.dataset1['street_type'].str.lower()
        self.dataset1['street_name'] = self.dataset1['street_name'].str.lower()
        self.dataset1['address_line2'] = self.dataset1['address_line2'].str.lower()
        self.dataset1['city'] = self.dataset1['city'].str.lower()

        self.dataset2['address'] = self.dataset2['address'].str.lower()
        self.dataset2['website'] = self.dataset2['website'].str.lower()
        self.dataset2['name'] = self.dataset2['name'].str.lower()

    @staticmethod
    def has_one_entry(matched_entries: pd.DataFrame) -> bool:
        """
        Checks whether only one entry exists.
        1st arg: Matched entry(ies) from a dataset.
        Returns: Boolean.
        """
        dframe = pd.DataFrame()
        dframe = matched_entries
        if dframe.shape[0] == 1:
            return True
        return False

    def search_target_dataset(self, id_code: int, snum: str,\
        sname: str, postal_code_city: str) -> dict:
        """
        Given an entry of a source dataset, a target dataset is searched.
        For each pair of entries, <source, target>, a matching score is computed.
        1st arg: The id from the entry of the source dataset.
        2nd arg: The street number from the entry of the source dataset.
        3rd arg: The street name from the entry of the source dataset.
        4th arg: The postal code and city from the entry of the source dataset.
        Returns: A dictionary containing the matches.
        Dictionary key: Id from source dataset.
        Dictionary value: A list where the 1st element is a matching score (1:absolute, <1: partial)
        and the 2nd element is the matched entry (dataframe) of target dataset.
        """
        source_id_code = id_code
        source_str_num = snum
        source_str_name = sname
        source_pc_city = postal_code_city.replace('-', ' ')

        if source_id_code in self.dataset2.id.values:
            hit_id = self.dataset2.loc[self.dataset2['id'] == source_id_code]
            match_score = 1.0
            self.matches[source_id_code] = [match_score, hit_id]
            self.full_matches += 1
            print("    ID in source entry found in target dataset:", source_id_code)
            print("      [Match score: ", match_score, "]")
        elif self.dataset2['address'].str.contains(source_pc_city, regex=False).any():
            hit_id = self.dataset2[self.dataset2['address'].str.contains(source_pc_city, regex=False)]
            street_num_name_re = re.escape(source_str_num) + " " + r"[a-zA-Z]+" +\
                " " + re.escape(source_str_name) + ","
            if hit_id['address'].str.contains(street_num_name_re, regex=True).any():
                hit_addr = hit_id[hit_id['address'].str.contains(street_num_name_re, regex=True)]
                if self.has_one_entry(hit_addr):
                    source_address = source_str_num + " " + source_str_name + " " + source_pc_city
                    target_str = hit_addr['address'].to_string()
                    match_score = self.estimate_match_score(source_address, target_str)
                    self.matches[source_id_code] = [match_score, hit_addr]
                    self.partial_matches = 1
                    print("    Address in source entry found in target dataset:", source_address)
                    print("      [Match score: ", match_score, "]")
        return self.matches

    def show_matches(self) -> None:
        """
        Reports (prints) the matched entries.
        Returns: None.
        """
        for mathched_id, matched_entry in self.matches.items():
            print("Matched id: ", mathched_id, " Matched entry: ")
            print(matched_entry)

    def report_match_stats(self) -> None:
        """
        Reports (prints) basic statistics of matches.
        Returns: None.
        """
        print("Num. of absolute matches (based on ID): ", self.full_matches)
        print("Num. of partial matches (based on address): ", self.partial_matches)

    def match_datasets(self) -> None:
        """
        Matches the entries of the two datasets.
        Returns: None:
        """
        for entry_counter, cur_entry in self.dataset1.iterrows():
            print("  Index of source entry under processing: ", entry_counter)
            postal_code_city = str(cur_entry.postal_code) + " " + cur_entry.city
            self.search_target_dataset(cur_entry.id, str(cur_entry.street_number), \
                cur_entry.street_name, postal_code_city)

    def evaluate_testonly(self) -> float:
        """
        Evaluates the matches.
        * Note: Given that no groundtruth exists,
        this is provided for testing purposes along with the evaluation metrics.
        Returns: F1 measure.
        """
        num_correct = 0
        precision = 0
        recall = 0
        for _, _ in self.matches.items():
            if random.choice([True, False]):
                num_correct += 1
        precision = num_correct / len(self.matches)
        recall = num_correct /  len(self.dataset1.index)
        self.evaluation_f1 = (2 * precision * recall) / (precision + recall)
        self.evaluation_f1 = ("%.2f" % self.evaluation_f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 measure: ", self.evaluation_f1)
        return self.evaluation_f1

    def run(self) -> None:
        """
        The main method: iterate over first dataset (aka source)
        searching forcmatches in the second dataset (aka target).
        Returns: None.
        """
        self.dataset1 = self.load_dataset(self.dataset1_filename)
        self.dataset2 = self.load_dataset(self.dataset2_filename)
        print("  Size of first dataset: ", self.dataset1.shape)
        print("  Size of second dataset: ", self.dataset2.shape)
        self.preprocess_datasets()
        print("  Size of first dataset after pre-processing: ", self.dataset1.shape)
        print("  Size of second dataset after pre-processing: ", self.dataset2.shape)
        self.match_datasets()


if __name__ == "__main__":
    DAT1 = "source1.tsv"
    DAT2 = "source2.tsv"

    lg.info("Matching: started.")
    MCHR = Matcher(DAT1, DAT2)
    MCHR.run()
    lg.info("Matching: completed.")

    lg.info("Display match statistics: started.")
    MCHR.report_match_stats()
    lg.info("Display match statistics: completed.")

    lg.info("Display matches: started.")
    MCHR.show_matches()
    lg.info("Display matches: completed.")

    lg.info("Evaluation (only for test): started.")
    MCHR.evaluate_testonly()
    lg.info("Evaluation (only for test): completed.")
