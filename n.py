"""Heap Sort with min heap, max heap, and heapify operations.

A heap is a complete binary tree stored in a list.
For an item at index i:
- left child: 2 * i + 1
- right child: 2 * i + 2
- parent: (i - 1) // 2
"""

def max_heapify(numbers, heap_size, index):
    """Restore the max-heap rule below index (parent >= children)."""
    largest = index
    left = 2 * index + 1
    right = 2 * index + 2

    if left < heap_size and numbers[left] > numbers[largest]:
        largest = left
    if right < heap_size and numbers[right] > numbers[largest]:
        largest = right

    if largest != index:
        numbers[index], numbers[largest] = numbers[largest], numbers[index]
        max_heapify(numbers, heap_size, largest)


def min_heapify(numbers, heap_size, index):
    """Restore the min-heap rule below index (parent <= children)."""
    smallest = index
    left = 2 * index + 1
    right = 2 * index + 2

    if left < heap_size and numbers[left] < numbers[smallest]:
        smallest = left
    if right < heap_size and numbers[right] < numbers[smallest]:
        smallest = right

    if smallest != index:
        numbers[index], numbers[smallest] = numbers[smallest], numbers[index]
        min_heapify(numbers, heap_size, smallest)


def build_max_heap(numbers):
    """Return a copy of numbers arranged as a max heap."""
    heap = numbers.copy()
    for index in range(len(heap) // 2 - 1, -1, -1):
        max_heapify(heap, len(heap), index)
    return heap


def build_min_heap(numbers):
    """Return a copy of numbers arranged as a min heap."""
    heap = numbers.copy()
    for index in range(len(heap) // 2 - 1, -1, -1):
        min_heapify(heap, len(heap), index)
    return heap


def heap_sort(numbers):
    """Return numbers in ascending order using a max heap."""
    heap = numbers.copy()
    heap_size = len(heap)
    for index in range(heap_size // 2 - 1, -1, -1):
        max_heapify(heap, heap_size, index)

    for end in range(heap_size - 1, 0, -1):
        heap[0], heap[end] = heap[end], heap[0]
        heap_size -= 1
        max_heapify(heap, heap_size, 0)
    return heap


def heap_sort_descending(numbers):
    """Return numbers in descending order using a min heap."""
    heap = numbers.copy()
    heap_size = len(heap)
    for index in range(heap_size // 2 - 1, -1, -1):
        min_heapify(heap, heap_size, index)

    for end in range(heap_size - 1, 0, -1):
        heap[0], heap[end] = heap[end], heap[0]
        heap_size -= 1
        min_heapify(heap, heap_size, 0)
    return heap


if __name__ == "__main__":
    values = [4, 10, 3, 5, 1, 8]
    print("Original:", values)
    print("Max heap:", build_max_heap(values))
    print("Min heap:", build_min_heap(values))
    print("Ascending:", heap_sort(values))
    print("Descending:", heap_sort_descending(values))
