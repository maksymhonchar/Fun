static final int KEYSARR_SIZE = 2000000;
static final int KEYS_RANGE = 4;

static void CountingSort(int[] arr, int range) {
   int[] keysCount = new int[range];
   for (int i = 0; i < arr.length; i++) {
      keysCount[arr[i]]++;
   }
   for (int j = 0, index = 0; j < keysCount.length; j++) {
      for (int k = 0; k < keysCount[j]; k++, index++) {
         arr[index] = j;
      }
   }
}
