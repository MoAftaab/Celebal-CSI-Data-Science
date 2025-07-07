def print_pyramid(rows):
    for i in range(rows):
        for j in range(rows - i - 1):
            print(" ", end=" ")
            
        for j in range(2 * i + 1):
            print("*", end=" ")
            
        print()  

if __name__ == "__main__":
    try:
        rows = int(input("Enter number of rows for the pyramid pattern: "))
        if rows <= 0:
            print("Please enter a positive integer")
        else:
            print("Pyramid Pattern:")
            print_pyramid(rows) 
    except ValueError:
        print("Invalid input. Please enter a valid integer.") 
