static final int KEYARR_MAX = 1000;
static final int KEY_RANGE = 5;

static int[] CountingSort_algo(int[] arr, int range) {
    int[] keysCount = new int[range];
    int[] result = new int[arr.length];
    for (int i = 0; i < arr.length; i++) {
        keysCount[arr[i]]++;
    }
    for (int i = 1; i < range; i++) {
        keysCount[i] += keysCount[i - 1];
    }
    for (int i = arr.length - 1; i >= 0; i--) {
        keysCount[arr[i]]--;
        result[keysCount[arr[i]]] = arr[i];
    }
    return (result);
}
