import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A simple task management system demonstrating classes, enums,
 * interfaces, generics, streams, and exception handling.
 */
public class TaskManager {

    public enum Priority {
        LOW, MEDIUM, HIGH, CRITICAL;

        public boolean isUrgent() {
            return this == HIGH || this == CRITICAL;
        }
    }

    public enum Status {
        TODO, IN_PROGRESS, DONE, CANCELLED
    }

    public static class Task implements Comparable<Task> {
        private final String title;
        private final Priority priority;
        private Status status;
        private final List<String> tags;

        public Task(String title, Priority priority) {
            if (title == null || title.isBlank()) {
                throw new IllegalArgumentException("Title cannot be null or blank");
            }
            this.title = title;
            this.priority = priority;
            this.status = Status.TODO;
            this.tags = new ArrayList<>();
        }

        public String getTitle() {
            return title;
        }

        public Priority getPriority() {
            return priority;
        }

        public Status getStatus() {
            return status;
        }

        public void setStatus(Status status) {
            if (this.status == Status.CANCELLED) {
                throw new IllegalStateException("Cannot change status of cancelled task");
            }
            this.status = status;
        }

        public void addTag(String tag) {
            if (!tags.contains(tag)) {
                tags.add(tag);
            }
        }

        public List<String> getTags() {
            return Collections.unmodifiableList(tags);
        }

        public boolean hasTag(String tag) {
            return tags.contains(tag);
        }

        @Override
        public int compareTo(Task other) {
            return Integer.compare(other.priority.ordinal(), this.priority.ordinal());
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof Task other)) return false;
            return title.equals(other.title) && priority == other.priority;
        }

        @Override
        public int hashCode() {
            return title.hashCode() * 31 + priority.hashCode();
        }

        @Override
        public String toString() {
            return String.format("[%s] %s (%s)", status, title, priority);
        }
    }

    private final List<Task> tasks;

    public TaskManager() {
        this.tasks = new ArrayList<>();
    }

    public Task addTask(String title, Priority priority) {
        Task task = new Task(title, priority);
        tasks.add(task);
        return task;
    }

    public void removeTask(String title) {
        tasks.removeIf(t -> t.getTitle().equals(title));
    }

    public List<Task> getTasksByPriority(Priority priority) {
        return tasks.stream()
                .filter(t -> t.getPriority() == priority)
                .collect(Collectors.toList());
    }

    public List<Task> getTasksByStatus(Status status) {
        return tasks.stream()
                .filter(t -> t.getStatus() == status)
                .collect(Collectors.toList());
    }

    public List<Task> getUrgentTasks() {
        return tasks.stream()
                .filter(t -> t.getPriority().isUrgent())
                .filter(t -> t.getStatus() != Status.DONE && t.getStatus() != Status.CANCELLED)
                .sorted()
                .collect(Collectors.toList());
    }

    public List<Task> searchByTag(String tag) {
        return tasks.stream()
                .filter(t -> t.hasTag(tag))
                .collect(Collectors.toList());
    }

    public int getTotalCount() {
        return tasks.size();
    }

    public int getCompletedCount() {
        return (int) tasks.stream()
                .filter(t -> t.getStatus() == Status.DONE)
                .count();
    }

    public double getCompletionRate() {
        if (tasks.isEmpty()) {
            return 0.0;
        }
        return (double) getCompletedCount() / getTotalCount() * 100.0;
    }

    public List<Task> getSortedTasks() {
        List<Task> sorted = new ArrayList<>(tasks);
        Collections.sort(sorted);
        return sorted;
    }

    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Tasks: %d total, %d completed (%.1f%%)\n",
                getTotalCount(), getCompletedCount(), getCompletionRate()));

        for (Priority p : Priority.values()) {
            long count = tasks.stream()
                    .filter(t -> t.getPriority() == p)
                    .count();
            if (count > 0) {
                sb.append(String.format("  %s: %d\n", p, count));
            }
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        TaskManager manager = new TaskManager();

        Task t1 = manager.addTask("Fix login bug", Priority.CRITICAL);
        t1.addTag("backend");
        t1.addTag("auth");

        Task t2 = manager.addTask("Update README", Priority.LOW);
        t2.addTag("docs");

        Task t3 = manager.addTask("Add unit tests", Priority.HIGH);
        t3.addTag("testing");
        t3.addTag("backend");

        Task t4 = manager.addTask("Refactor database layer", Priority.MEDIUM);
        t4.addTag("backend");
        t4.addTag("database");

        t2.setStatus(Status.DONE);
        t1.setStatus(Status.IN_PROGRESS);

        System.out.println("=== Task Manager Demo ===\n");
        System.out.println(manager.getSummary());

        System.out.println("Urgent tasks:");
        for (Task t : manager.getUrgentTasks()) {
            System.out.println("  " + t);
        }

        System.out.println("\nBackend tasks:");
        for (Task t : manager.searchByTag("backend")) {
            System.out.println("  " + t);
        }

        System.out.println("\nAll tasks (sorted by priority):");
        for (Task t : manager.getSortedTasks()) {
            System.out.println("  " + t);
        }
    }
}
