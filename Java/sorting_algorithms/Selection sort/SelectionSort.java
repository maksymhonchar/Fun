import java.util.Calendar;
import java.util.Random;

public class SelectionSort {

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

	static long selectionSort(int[] arr) {
		long swaps = 0;
		for (int i = 0; i < arr.length - 1; i++) {
			int minIndex = i;
			for (int j = i + 1; j < arr.length; j++) {
				if (arr[j] < arr[minIndex]) {
					minIndex = j;
				}
			}
			swapInt(arr, i, minIndex);
			swaps++;
		}
		return (swaps);
	}

	static long selectionSortSedjwik(int[] arr) {
		long swaps = 0;
		for (int i = 0; i < arr.length; i++) {
			int minIndex = i;
			for (int j = i + 1; j < arr.length; j++) {
				if (arr[j] < arr[minIndex]) {
					minIndex = j;
				}
			}
			swapInt(arr, i, minIndex);
			swaps++;
		}
		return (swaps);
	}

	static long selectionSort_CheckBeforeSwap(int[] arr) {
		long swaps = 0;
		for (int i = 0; i < arr.length - 1; i++) {
			int minIndex = i;
			for (int j = i + 1; j < arr.length; j++) {
				if (arr[j] < arr[minIndex]) {
					minIndex = j;
				}
			}
			if(i != minIndex) {
				swapInt(arr, i, minIndex);
				swaps++;
			}
		}

		return (swaps);
	}

	public static void main(String[] args) {
		// A standart testing array. Never change his items.
		final int TEST_SIZE = 500000;
		int[] testArr = new int[TEST_SIZE];
		fillArray(testArr);

		// ---------------------------------------
		// STANDART SELECTION SORT
		// ---------------------------------------
		int[] Arr_standartSelection = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_standartSelection, 0, testArr.length);
		// calculate time before sorting
		double beforeSort = getCurrentTime();
		// do a sorting
		long swaps_standart = selectionSort(Arr_standartSelection);
		// calculate time after sorting
		double afterSort = getCurrentTime();
		// print info
		System.out.println("Standart selection sort:");
		System.out.println("Swaps used: " + swaps_standart);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

		// ---------------------------------------
		// STANDART SELECTION SORT
		// ---------------------------------------
		int[] Arr_selectionSedgwick = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_selectionSedgwick, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		long swaps_selectionSedgwick = selectionSortSedjwik(Arr_selectionSedgwick);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("Sedgwick selection sort:");
		System.out.println("Swaps used: " + swaps_selectionSedgwick);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

		// ---------------------------------------
		// SELECTION SORT - CHECK BEFORE SWAP
		// ---------------------------------------
		int[] Arr_betterSelection = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_betterSelection, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		long swaps_betterSelection = selectionSort_CheckBeforeSwap(Arr_betterSelection);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("Standart selection sort:");
		System.out.println("Swaps used: " + swaps_betterSelection);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

	}

}
