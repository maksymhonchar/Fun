import java.util.Calendar;
import java.util.Random;

public class InsertionSort {

	static void swapInt(int[] arr, int aIndex, int bIndex) {
		int temp = arr[aIndex];
		arr[aIndex] = arr[bIndex];
		arr[bIndex] = temp;
	}

	static void fillArray(int[] arr) {
		Random r = new Random();
		for (int i = 0; i < arr.length; i++)
			arr[i] = r.nextInt(1000);
	}

	static double getCurrentTime() {
		double currentTime_hours = Calendar.getInstance().get(Calendar.HOUR);
		double currentTime_minutes = Calendar.getInstance().get(Calendar.MINUTE);
		double currentTime_seconds = Calendar.getInstance().get(Calendar.SECOND);
		double currentTime_milliseconds = (double) Calendar.getInstance().get(Calendar.MILLISECOND) / 1000;
		return (currentTime_seconds + currentTime_milliseconds + 60 * currentTime_minutes + 3600 * currentTime_hours);
	}

	static long insertionSort_Sedgewick(int[] arr) {
		long swaps = 0;
		for (int i = 1; i < arr.length; i++) {
			for (int j = i; j > 0 && arr[j] < arr[j - 1]; j--) {
				swapInt(arr, j, j - 1);
				swaps++;
			}
		}
		return (swaps);
	}

	static long insertionSort_MIT(int[] arr) {
		long swaps = 0;
		for (int i = 1; i < arr.length; i++) {
			int elem = arr[i];
			int j = i - 1;
			while (j >= 0 && arr[j] > elem) {
				arr[j + 1] = arr[j];
				j--;
			}
			arr[j + 1] = elem;
			swaps++;
		}
		return (swaps);
	}

	public static void main(String[] args) {
		// A standart testing array. Never change his items.
		final int TEST_SIZE = 10000;
		int[] testArr = new int[TEST_SIZE];
		fillArray(testArr);

		// ---------------------------------------
		// SEDGEWICK INSERTION SORT
		// ---------------------------------------
		int[] Arr_insertionSedgewick = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_insertionSedgewick, 0, testArr.length);
		// calculate time before sorting
		double beforeSort = getCurrentTime();
		// do a sorting
		long swaps_insertionSedgewick = insertionSort_Sedgewick(Arr_insertionSedgewick);
		// calculate time after sorting
		double afterSort = getCurrentTime();
		// print info
		System.out.println("Sedgewick insertion sort:");
		System.out.println("Swaps used: " + swaps_insertionSedgewick);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

		// ---------------------------------------
		// MIT INSERTION SORT
		// ---------------------------------------
		int[] Arr_insertionSort_MIT = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_insertionSort_MIT, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		long swaps_insertionMIT = insertionSort_MIT(Arr_insertionSort_MIT);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("MIT insertion sort:");
		System.out.println("Swaps used: " + swaps_insertionMIT);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));
	}
}
