class RulerDrawer(object):
    """Class to represent process of Ruler drawing problem."""

    def draw_line(self, tick_length, tick_label=''):
        """Draw one line with given tick length (followed by optional label)."""
        line = '-' * tick_length
        if tick_label:
            line += ' ' + tick_label
        print(line)

    def draw_interval(self, center_length):
        """Draw tick interval based upon a central tick length."""
        if center_length > 0:
            self.draw_interval(center_length - 1)
            self.draw_line(center_length)
            self.draw_interval(center_length - 1)

    def draw_ruler(self, num_inches, major_length):
        """Draw ruler with given number of inches, major tick length."""
        self.draw_line(major_length, '0')
        for j in range(1, 1+num_inches):
            self.draw_interval(major_length - 1)
            self.draw_line(major_length, str(j))

if __name__ == "__main__":
    rd = RulerDrawer()
    rd.draw_ruler(5, 3)
