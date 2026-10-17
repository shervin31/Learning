class book():
    def __init__(self, title, author):
        self.title = title
        self.author = author


class library():
    def __init__(self, books: list, name: str, city: str):
        self.books = list()
        self.name = name
        self.city = city
    
    def add_books_to_library(self, books: list):
        for book in books: 
            self.books.append(book)

    

def main():
    fairview = library([],'Fairview Library', 'Toronto')

    moby_dick = book('Moby Dick', 'Darwin')

    fairview.add_books_to_library([moby_dick])

    print(fairview.books[0].title)

main()
    

    
    