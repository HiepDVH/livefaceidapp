import argparse

from recognize import extract_feature_dict

parser = argparse.ArgumentParser(description='Extract face features in a face database')

parser.add_argument('--database_path', help='Path to database')
parser.add_argument('--feature_path', help='Path to feature file where feature will be saved')

args = parser.parse_args()
if __name__ == '__main__':
    extract_feature_dict(
        database_path=args.database_path,
        feature_path=args.feature_path,
    )
