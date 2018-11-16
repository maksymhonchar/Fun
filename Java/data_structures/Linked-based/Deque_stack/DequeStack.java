public class DequeStack {
	private Linked_Deque D;

	public DequeStack() {
		this.D = new Linked_Deque();
	}

	public int size() {
		return (this.D.size());
	}
	public boolean empty() {
		return (this.D.empty());
	}
	
	public void push(int element) {
		this.D.insertLast(element);
	}
	
	public int peek() {
		if(this.D.empty()) {
			return (0);
		}
		return (this.D.last());
	}
	public int pop() {
		if(empty()) {
			return (0);
		}
		return (this.D.removeLast());						
	}
}
