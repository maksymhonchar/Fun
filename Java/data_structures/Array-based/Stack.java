public class Stack {
	private static final int MAX_CAPACITY = 1000;
	private int stackArr[];
	private int top;

	public Stack() {
		this.stackArr = new int[MAX_CAPACITY];
		this.top = -1;
	}

	public Stack(int cap) {
		this.stackArr = new int[cap];
		this.top = -1;
	}

	public int size() {
		return (top + 1);
	}

	public boolean empty() {
		return (top < 0);
	}

	public int peek() {
		if (empty()) {
			return (0);
		}
		return (stackArr[top]);
	}

	public void push(int element) {
		if (size() == stackArr.length) {
			return;
		}
		top++;
		stackArr[top] = element;
	}

	public int pop() {
		if (empty()) {
			return (0);
		}
		int temp = stackArr[top];
		stackArr[top] = 0;
		top--;
		return (temp);
	}

	/**
	 * Testing function. <br>
	 * Displays all the elements of the stack.
	 */
	public void print() {
		if (empty()) {
			System.out.println("Stack is empty.");
			return;
		}
		for (int i = 0; i <= top; i++) {
			System.out.printf("%d ", stackArr[i]);
		}
		System.out.println("");
	}
}
