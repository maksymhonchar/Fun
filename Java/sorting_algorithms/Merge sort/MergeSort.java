import java.util.*;

public class MergeSort {

	static void fillArray(Integer[] arr) {
		Random r = new Random();
		for (int i = 0; i < arr.length; i++)
			arr[i] = (Integer)r.nextInt(1000);
	}

	static double getCurrentTime() {
		double currentTime_hours = Calendar.getInstance().get(Calendar.HOUR);
		double currentTime_minutes = Calendar.getInstance().get(Calendar.MINUTE);
		double currentTime_seconds = Calendar.getInstance().get(Calendar.SECOND);
		double currentTime_milliseconds = (double) Calendar.getInstance().get(Calendar.MILLISECOND) / 1000;
		return (currentTime_seconds + currentTime_milliseconds + 60 * currentTime_minutes + 3600 * currentTime_hours);
	}

	@SuppressWarnings("rawtypes")
	static Comparable[] mergeSort(Comparable[] list) {
		if (list.length <= 1) {
			return (list);
		}
		// Divide
		Comparable[] first = new Comparable[list.length / 2];
		Comparable[] second = new Comparable[list.length - first.length];
		System.arraycopy(list, 0, first, 0, first.length);
		System.arraycopy(list, first.length, second, 0, second.length);
		// Recursion
		mergeSort(first);
		mergeSort(second);
		// Conquer
		merge(first, second, list);
		return (list);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	static void merge(Comparable[] first, Comparable[] second, Comparable[] result) {
		int indexFirst = 0;
		int indexSecond = 0;
		int indexMerged = 0;
		// Comparison
		while (indexFirst < first.length && indexSecond < second.length) {
			if (first[indexFirst].compareTo(second[indexSecond]) < 0) {
				result[indexMerged] = first[indexFirst];
				indexFirst++;
			} else {
				result[indexMerged] = second[indexSecond];
				indexSecond++;
			}
			indexMerged++;
		}
		// Copy REMAINING elements here!
		System.arraycopy(first, indexFirst, result, indexMerged, first.length - indexFirst);
		System.arraycopy(second, indexSecond, result, indexMerged, second.length - indexSecond);
	}

	public static void main(String[] args) {
		// A standart testing array. Never change his items.
		final int TEST_SIZE = 5000000;
		Integer[] testArr = new Integer[TEST_SIZE];
		fillArray(testArr);
		
		// ---------------------------------------
		// RECURSIVE MERGE SORT
		// ---------------------------------------
		Integer[] Arr_mergeSort = new Integer[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_mergeSort, 0, testArr.length);
		// calculate time before sorting
		double beforeSort = getCurrentTime();
		// do a sorting
		mergeSort(Arr_mergeSort);
		// calculate time after sorting
		double afterSort = getCurrentTime();
		// print info
		System.out.println("Recursive merge sort.");
		System.out.println("Array to sort: " + TEST_SIZE + " elements.");
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));
	}

}
