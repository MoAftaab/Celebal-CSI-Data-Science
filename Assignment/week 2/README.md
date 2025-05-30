# Singly Linked List Implementation

## Overview
This project implements a singly linked list data structure using Object-Oriented Programming principles in Python. The implementation follows best practices for Python programming, including proper documentation, exception handling, and code organization.

## Files
- `linked_list.py`: Contains the complete implementation of the linked list data structure

## Features
The implementation includes:

1. A `Node` class to represent individual nodes in the linked list
2. A `LinkedList` class with the following methods:
   - `append(data)`: Add a new node to the end of the list
   - `print_list()`: Display all elements in the list
   - `delete_nth(n)`: Delete the nth node (1-based indexing)
3. Exception handling for edge cases:
   - Attempting to delete from an empty list
   - Attempting to delete a node with an index out of range

## Diagrams

### Basic Structure of a Singly Linked List

```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 1 │ ●─┼───>│ 2 │ ●─┼───>│ 3 │ ●─┼───>│ 4 │ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
  Node 1       Node 2       Node 3       Node 4
```

### Append Operation (Adding a New Node)

Before adding node 5:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 1 │ ●─┼───>│ 2 │ ●─┼───>│ 3 │ ●─┼───>│ 4 │ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
```

After adding node 5:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 1 │ ●─┼───>│ 2 │ ●─┼───>│ 3 │ ●─┼───>│ 4 │ ●─┼───>│ 5 │ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
```

### Delete Operation (Removing the 3rd Node)

Before deletion:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐
│ 1 │ ●─┼───>│ 2 │ ●─┼───>│ 3 │ ●─┼───>│ 4 │ / │
└───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘
```

After deleting the 3rd node:
```
head
 │
 ▼
┌───┬───┐    ┌───┬───┐             ┌───┬───┐
│ 1 │ ●─┼───>│ 2 │ ●─┼─────────────>│ 4 │ / │
└───┴───┘    └───┴───┘             └───┴───┘
                  │                    ▲
                  │    ┌───┬───┐       │
                  └────┤ 3 │ ●─┼───────┘
                       └───┴───┘
                      (disconnected)
```

## How to Run
To run the program, use Python 3:

```
python linked_list.py
```

## Example Output
When you run the program, you should see output similar to:

```
Original linked list:
10 -> 20 -> 30 -> 40 -> 50

Linked list after deleting 3rd node:
10 -> 20 -> 40 -> 50

Error: Index 10 is out of range

Deleting all nodes one by one:
20 -> 40 -> 50
40 -> 50
50
Error: Cannot delete from an empty list
```

## Implementation Details

### Node Class
Each node in the linked list contains:
- `data`: The value stored in the node
- `next`: A reference to the next node in the list (or None if it's the last node)

```
┌─────────────┐
│    Node     │
├─────────────┤
│ data        │  <- The value stored in the node
│ next        │  <- Reference to the next node
└─────────────┘
```

### LinkedList Class
The linked list is implemented with a reference to the head node and provides methods for common operations:
- Adding elements (at the end)
- Traversing and printing the list
- Deleting elements at a specific position

```
┌────────────────┐
│   LinkedList   │
├────────────────┤
│ head           │  <- Reference to the first node
├────────────────┤
│ append()       │  <- Add a node at the end
│ print_list()   │  <- Display all elements
│ delete_nth()   │  <- Delete the nth node
└────────────────┘
```

The implementation handles edge cases properly with appropriate exception handling. 