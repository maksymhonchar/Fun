
public static void bucketSort(int[] arr) {
    Queue<Integer>[] buckets = new Queue[10];
    for (int i = 0; i < 10; i++)
        buckets[i] = new LinkedList<Integer>();
    boolean sorted = false;
    int expo = 1;
    while (!sorted) {
        sorted = true;
        for (int item : arr) {
            int bucket = (item / expo) % 10;
            if (bucket > 0)
                sorted = false;
            buckets[bucket].add(item);
        }
        expo *= 10;
        int index = 0;
        for (Queue<Integer> bucket : buckets)
            while (!bucket.isEmpty())
                arr[index++] = bucket.remove();
    }
}

private static boolean isSorted(int[] arr) {
    for (int i = 1; i < arr.length; i++)
        if (arr[i - 1] > arr[i])
            return (false);
    return (true);
}