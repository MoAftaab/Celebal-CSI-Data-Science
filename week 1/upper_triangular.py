def print_upper_triangular(rows):
    for i in range(rows, 0, -1):
        for j in range(i):
            print("*", end=" ")
        print() 

if __name__ == "__main__":
    try:
        rows = int(input("Enter number of rows for the upper triangular pattern: "))
        if rows <= 0:
            print("Please enter a positive integer")
        else:
            print("Upper Triangular Pattern:")
            print_upper_triangular(rows)
    except ValueError:
        print("Invalid input. Please enter a valid integer.") 
