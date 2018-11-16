public class LinkedStack {
	private Node top;
	private int size;

	public LinkedStack() {
		this.top = null;
		this.size = 0;
	}

	public int size() {
		return (this.size);
	}

	public boolean empty() {
		return (null == this.top);
	}

	public void push(int element) {
		Node n = new Node();
		n.setElement(element);
		n.setNext(this.top);
		this.top = n;
		this.size++;
	}

	public int peek() {
		if (empty()) {
			return (0);
		}
		return (this.top.getElement());
	}

	public int pop() {
		if (empty()) {
			return (0);
		}
		int temp = this.top.getElement();
		this.top = this.top.getNext();
		size--;
		return (temp);
	}
}
