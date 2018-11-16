public class Node {
	private int element;
	private Node next;

	public Node() {
		this(0, null);
	}

	public Node(int e, Node n) {
		this.element = e;
		this.next = n;
	}

	public int getElement() {
		return (this.element);
	}

	public Node getNext() {
		return (this.next);
	}

	void setElement(int element) {
		this.element = element;
	}

	void setNext(Node next) {
		this.next = next;
	}

}
