public static void radixSort_LSD(int[] a) {
    for (int d = 1; d <= (int) Math.pow(10, 3); d *= 10) { //3 in math.pow means there are 3 digits in number (i.e. 123 or 522)
        int[] aux = countintSortForLSD(a, d);
        for (int i = 0; i < a.length; i++) {
            a[i] = aux[i];
        }
    }
}

private static int[] countintSortForLSD(int[] inp, int pos) {
    int[] out = new int[inp.length];
    int[] count = new int[10]; // for digits 0-9
    for (int i = 0; i < inp.length; i++) {
        int digit = (inp[i] / pos) % 10;
        count[digit]++;
    }
    for (int i = 1; i < count.length; i++) {
        count[i] += count[i - 1];
    }
    for (int i = inp.length; i >= 0; i--) {
        int digit = (inp[i] / pos) % 10; //example: pos=10 (second) inp[i]=123. Result will be (123/10)%10 = 2 - a second digit
        count[digit]--;
        out[count[digit]] = inp[i];
    }
    return (out);
}
