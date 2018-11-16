import java.util.Calendar;
import java.util.Random;

public class BubbleSort {

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

	static long bubbleSort(int[] arr) {
		int n = arr.length;
		long swaps = 0;
		for (int i = 0; i < n - 1; i++) {
			for (int j = 0; j < n - 1 - i; j++) {
				if (arr[j + 1] < arr[j]) {
					swapInt(arr, j, j + 1);
					swaps++;
				}
			}
		}
		return (swaps);
	}

	static long betterBubbleSort(int[] arr) {
		int n = arr.length;
		long swaps = 0;
		boolean swapped = true;
		for (int i = 0; i < n - 1; i++) {
			swapped = false;
			for (int j = 0; j < n - 1 - j; j++) {
				if (arr[j] > arr[j + 1]) {
					swapInt(arr, j, j + 1);
					swaps++;
					swapped = true;
				}
			}
		}
		return (swaps);
	}

	static long shakerBubbleSort(int[] arr) {
		long swaps = 0;
		int l = 0;
		int r = arr.length - 1;
		do {
			for (int i = 1; i < r; i++) {
				if (arr[i + 1] < arr[i]) {
					swapInt(arr, i + 1, i);
					swaps++;
				}
			}
			r--;
			for (int i = r; i > 1; i--) {
				if (arr[i] < arr[i - 1]) {
					swapInt(arr, i, i - 1);
					swaps++;
				}
			}
			l++;
		} while (l < r);
		return (swaps);
	}

	public static void main(String[] args) {
		//A standart testing array. Never change his items.
		final int TEST_SIZE = 10000;
		int[] testArr = new int[TEST_SIZE];
		fillArray(testArr);
		
		// ---------------------------------------
		// STANDART BUBBLE SORT
		// ---------------------------------------
		int[] Arr_standartBubbleSort = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_standartBubbleSort, 0, testArr.length);
		// calculate time before sorting
		double beforeSort = getCurrentTime();
		// do a sorting
		long swaps_standart = bubbleSort(Arr_standartBubbleSort);
		// calculate time after sorting
		double afterSort = getCurrentTime();
		// print info
		System.out.println("Standart bubble sort:");
		System.out.println("Swaps used: " + swaps_standart);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));
		
		// ---------------------------------------
		// BETTER BUBBLE SORT - WITH BOOLEAN CHECK
		// ---------------------------------------
		int[] Arr_betterBubbleSort = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_betterBubbleSort, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		long swaps_better = betterBubbleSort(Arr_betterBubbleSort);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("Better bubble sort with boolean:");
		System.out.println("Swaps used: " + swaps_better);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));
		
		// ---------------------------------------
		// TWO-FORKED BUBBLE SORT - SHAKER SORT
		// ---------------------------------------
		int[] Arr_shakerBubbleSort = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_shakerBubbleSort, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		long swaps_shaker = shakerBubbleSort(Arr_shakerBubbleSort);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("Shaker bubble sort:");
		System.out.println("Swaps used: " + swaps_shaker);
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

	}

}
