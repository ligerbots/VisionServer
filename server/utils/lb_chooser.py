#!/usr/bin/python3

# Class to create a SendableChooser-like thing in NetworkTables
# Don't use the official one for the vision coprocessor.
#  That pulls in the whole "wpilib" package for just this. Gross overkill.

from networktables import NetworkTables


class LB_Chooser:
    '''LigerBots NT Chooser class. This is lighterweight than the WPILib class'''

    def __init__(self, key, ntinst=None):
        if ntinst is None:
            self._ntinst = NetworkTables
        else:
            self._ntinst = ntinst

        self.base_key = '/SmartDashboard/' + key

        # build the "directory" structure
        self._ntinst.getEntry(self.base_key + '/.controllable').setBoolean(True)
        self._ntinst.getEntry(self.base_key + '/.instance').setDouble(0.0) # not sure if this needs to be different
        self._ntinst.getEntry(self.base_key + '/.name').setString(key)
        self._ntinst.getEntry(self.base_key + '/.type').setString('String Chooser')
        self._ntinst.getEntry(self.base_key + '/active').setString('')
        self._ntinst.getEntry(self.base_key + '/default').setString('')
        self._ntinst.getEntry(self.base_key + '/options').setStringArray([])
        self._ntinst.getEntry(self.base_key + '/selected').setString('')
        self._selected = self._ntinst.getEntry(self.base_key + '/selected')
        self._selected.setString('')

        return

    def get_selected(self):
        return self._selected.getString('')

    def set_selected(self, value):
        return self._selected.setString(value)

    def set_default(self, value):
        '''Set the default choice'''

        self._ntinst.getEntry(self.base_key + '/default').setString(value)
        return

    def add_choice(self, value):
        '''Add an option to the list of choices'''

        entry = self._ntinst.getEntry(self.base_key + '/options')
        curr_choices = entry.getStringArray([])
        if value not in curr_choices:
            curr_choices = list(curr_choices)
            curr_choices.append(value)
        entry.setStringArray(curr_choices)
        return

    def add_choices(self, value_array):
        '''Add an option to the list of choices'''

        entry = self._ntinst.getEntry(self.base_key + '/options')
        curr_choices = entry.getStringArray([])
        curr_choices.extend(value_array)
        entry.setStringArray(curr_choices)
        return


if __name__ == '__main__':
    # test cases
    import time

    NetworkTables.startServer()

    chooser = LB_Chooser('vision/active_mode')
    chooser.add_choices(('a', 'b'))

    old_choice = None
    while True:
        c = chooser.get_selected()
        if c != old_choice:
            print('choice', c)
        old_choice = c
        time.sleep(1)
