public class List {
	private final int STANDART_CAPACITY = 1000;
	private int[] list;
	private int size;

	public List() {
		this.list = new int[STANDART_CAPACITY];
		this.size = 0;
	}

	public List(int capacity) {
		this.list = new int[capacity];
		this.size = 0;
	}

	public boolean full() {
		return (list.length == size);
	}

	public boolean empty() {
		return (size == 0);
	}

	public boolean add(int element) {
		if (full()) {
			return (false);
		}
		list[size] = element;
		size++;
		return (true);
	}

	public boolean addByIndex(int index, int element) {
		if (full()) {
			return (false);
		}
		for (int i = size; i >= index; i--) {
			list[i] = list[i - 1];
		}
		list[index] = element;
		size++;
		return (true);
	}

	public int removeByIndex(int index) {
		if (empty()) {
			return (0);
		}
		int temp = list[index];
		for (int i = index; i < size; i++) {
			list[i] = list[i + 1];
		}
		list[size] = 0;
		size--;
		return (temp);
	}

	public boolean remove(int element) {
		if (empty()) {
			return (false);
		}
		for (int i = 0; i < size; i++) {
			if (list[i] == element) {
				removeByIndex(i);
				return (true);
			}
		}
		return (false);
	}

	public int getByIndex(int index) {
		if (index >= size) {
			return (0);
		}
		return (list[index]);
	}

	public void setByIndex(int index, int element) {
		if (index >= size) {
			return;
		}
		list[index] = element;
	}

	/**
	 * Testing function. <br/>
	 * Prints the list out.
	 */
	public void print() {
		if (empty()) {
			System.out.println("List is empty.");
			return;
		}
		for (int i = 0; i < size; i++)
			System.out.printf("%d ", list[i]);
		System.out.println("");
	}
}
