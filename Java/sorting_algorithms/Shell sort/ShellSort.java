import java.util.Calendar;
import java.util.Random;

public class ShellSort {

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

	static void shellSort_ShellSeq(int[] arr) {
		for (int step = arr.length / 2; step > 0; step /= 2) {
			for (int i = step; i < arr.length; i++) {
				int temp = arr[i];
				int j = i;
				for (; j >= step && temp < arr[j - step]; j -= step) {
					arr[j] = arr[j - step];
				}
				arr[j] = temp;
			}
		}
	}

	static void shellSort_KnuthSeq(int[] arr) {
		// get the sequence
		int h = 1;
		while (h < arr.length / 3) {
			h = h * 3 + 1;
		}
		while (h > 0) {
			for (int i = h; i < arr.length; i++) {
				int temp = arr[i];
				int j = i;
				while (j > h - 1 && arr[j - h] >= temp) {
					arr[j] = arr[j - h];
					j -= h;
				}
				arr[j] = temp;
			}
			h /= 3;
		}
	}

	public static void main(String[] args) {
		// A standart testing array. Never change his items.
		final int TEST_SIZE = 50000;
		int[] testArr = new int[TEST_SIZE];
		fillArray(testArr);

		// ---------------------------------------
		// SHELL SORT - SEQUENCE BY SHELL
		// ---------------------------------------
		int[] Arr_shellSeq = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_shellSeq, 0, testArr.length);
		// calculate time before sorting
		double beforeSort = getCurrentTime();
		// do a sorting
		shellSort_ShellSeq(Arr_shellSeq);
		// calculate time after sorting
		double afterSort = getCurrentTime();
		// print info
		System.out.println("Shell sort with sequence N/2->N/4->N/8...1");
		System.out.println("Array to sort: " + TEST_SIZE + " elements.");
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));

		// ---------------------------------------
		// SEDWICK SHELL SORT - SEQUENCE BY KNUTH
		// ---------------------------------------
		int[] Arr_knuthSeq = new int[TEST_SIZE];
		System.arraycopy(testArr, 0, Arr_knuthSeq, 0, testArr.length);
		// calculate time before sorting
		beforeSort = getCurrentTime();
		// do a sorting
		shellSort_KnuthSeq(Arr_knuthSeq);
		// calculate time after sorting
		afterSort = getCurrentTime();
		// print info
		System.out.println("Shell sort with Knuth sequence (3^k - 1)/2 -> 1,3,13,40,121...");
		System.out.println("Array to sort: " + TEST_SIZE + " elements.");
		System.out.printf("Time spent: %.3f s\n\n", (afterSort - beforeSort));
	}
}
