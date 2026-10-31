# dùng scikit-learn cho Logistic Regression và Softmax Regression để so sánh với bản numpy ở asm1/asm2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import prepare_data, CLASS_NAMES_MULTI

def main():
    # Logistic Regression
    X_train, X_test, y_train, y_test, _ = prepare_data("binary", "cars_raw.csv")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print("== sklearn Logistic Regression (xe rẻ / đắt) - Test ==")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["rẻ", "đắt"], digits=4))

    # Softmax Regression
    # với bài toán đa lớp, LogisticRegression của sklearn mặc định dùng hàm softmax (multinomial)
    X_train, X_test, y_train, y_test, _ = prepare_data("multi", "cars_raw.csv")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print("== sklearn Softmax Regression (4 phân khúc giá) - Test ==")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=CLASS_NAMES_MULTI, digits=4))

if __name__ == "__main__":
    main()
