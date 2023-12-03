import random
import argparse

def naver_shopping(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    review_list = []
    for raw_text in lines:
        label, review = raw_text.strip().replace(",", "").split('\t')  # tab으로 구분돼있는 별점과 리뷰를 분할
        label = str(int(int(label) > 3))  # 4점 이상이면 긍정 (1) , 3점 이하면 부정 (0)
        review_list.append(label + "," + review)

    random.shuffle(review_list)

    train_idx = int(len(review_list) * 0.8)
    val_idx = int(len(review_list) * 0.1)

    with open("C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/naver_shopping_train.csv", "w", encoding="utf-8") as f:
        f.write("label,sentence1\n")
        for label_review in review_list[:train_idx]:
            f.write(label_review + "\n")

    with open("C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/naver_shopping_dev.csv", "w", encoding="utf-8") as f:
        f.write("label,sentence1\n")
        for label_review in review_list[train_idx:-val_idx]:
            f.write(label_review + "\n")

    with open("C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/naver_shopping_test.csv", "w", encoding="utf-8") as f:
        f.write("label,sentence1\n")
        for label_review in review_list[-val_idx:]:
            f.write(label_review + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="naver")
    args = parser.parse_args()

    if args.mode == "naver":
        naver_shopping("C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/naver_shopping.tsv")

if __name__ == "__main__":
    main()