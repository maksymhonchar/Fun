static final int MAX_STUDENTS = 10;

// Hello!
// So with this thing algorithm will sort FIRST by Faculty, and THEN by group. 
static int StudentsCompareFacultyGroup(Student s1, Student s2) {
    int result = s1.getFaculty().compareTo(s2.getFaculty());
    if (result == 0) {
        return Integer.valueOf(s1.getGroup()).compareTo(s2.getGroup());
    } else {
        return result;
    }
}

// Ayy lmao
static void InsertionSort(Student[] arr) {
    for (int i = 1; i < arr.length; i++) {
        Student elem = arr[i];
        int j = i - 1;
        while (j >= 0 && StudentsCompareFacultyGroup(arr[j], elem) > 0) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = elem;
    }
}