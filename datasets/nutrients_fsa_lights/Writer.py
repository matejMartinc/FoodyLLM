import csv


class Writer:

    @staticmethod
    def write(save_file: str, append: bool, questions: list, answers: list):
        if append:
            with open(save_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\n')
                for question, answer in zip(questions, answers):
                    writer.writerow([question])
                    writer.writerow([answer])
        else:
            with open(save_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\n')
                for question, answer in zip(questions, answers):
                    writer.writerow([question])
                    writer.writerow([answer])
