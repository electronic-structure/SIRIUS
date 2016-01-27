__copyright = """
PYCIFRW License Agreement (Python License, Version 2)
-----------------------------------------------------

1. This LICENSE AGREEMENT is between the Australian Nuclear Science
and Technology Organisation ("ANSTO"), and the Individual or
Organization ("Licensee") accessing and otherwise using this software
("PyCIFRW") in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement,
ANSTO hereby grants Licensee a nonexclusive, royalty-free, world-wide
license to reproduce, analyze, test, perform and/or display publicly,
prepare derivative works, distribute, and otherwise use PyCIFRW alone
or in any derivative version, provided, however, that this License
Agreement and ANSTO's notice of copyright, i.e., "Copyright (c)
2001-2014 ANSTO; All Rights Reserved" are retained in PyCIFRW alone or
in any derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates PyCIFRW or any part thereof, and wants to make the
derivative work available to others as provided herein, then Licensee
hereby agrees to include in any such work a brief summary of the
changes made to PyCIFRW.

4. ANSTO is making PyCIFRW available to Licensee on an "AS IS"
basis. ANSTO MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, ANSTO MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYCIFRW WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. ANSTO SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYCIFRW
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A
RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYCIFRW, OR ANY
DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between ANSTO
and Licensee. This License Agreement does not grant permission to use
ANSTO trademarks or trade name in a trademark sense to endorse or
promote products or services of Licensee, or any third party.

8. By copying, installing or otherwise using PyCIFRW, Licensee agrees
to be bound by the terms and conditions of this License Agreement.

"""


from urllib import *         # for arbitrary opening
import re
import copy
# For Python 2.6 or higher compatibility
try: 
    set
except NameError:
    import sets
    set = sets.Set

class StarList(list):
    pass

class StarDict(dict):
    pass


class LoopBlock(object):
    def __init__(self,parent_block,dataname):
        self.loop_no = parent_block.FindLoop(dataname)
        if self.loop_no < 0:
            raise KeyError, '%s is not in a loop structure' % dataname
        self.parent_block = parent_block

    def keys(self):
        return self.parent_block.loops[self.loop_no]

    def values(self):
        return [self.parent_block[a] for a in self.keys()]

    def items(self):
        return zip(self.keys(),self.values())

    def __getitem__(self,dataname):
        if isinstance(dataname,int):   #a packet request
            return self.GetPacket(dataname)
        if dataname in self.keys():
            return self.parent_block[dataname]
        else:
            raise KeyError, '%s not in loop block' % dataname

    def __setitem__(self,dataname,value):
        self.parent_block[dataname] = value
        self.parent_block.AddLoopName(self.keys()[0],dataname)

    def has_key(self,key):
        return key in self.parent_block.loops[self.loop_no]

    def __iter__(self):
        packet_list = zip(*self.values())
        names = self.keys()
        for p in packet_list:
            r = StarPacket(p)
            for n in range(len(names)):
                setattr(r,names[n].lower(),r[n])
            yield r

    # for compatibility
    def __getattr__(self,attname):
        return getattr(self.parent_block,attname)

    def load_iter(self,coords=[]):
        count = 0        #to create packet index 
        while not self.popout:
            # ok, we have a new packet:  append a list to our subloops
            for aloop in self.loops:
                aloop.new_enclosing_packet()
            for iname in self.item_order:
                if isinstance(iname,LoopBlock):       #into a nested loop
                    for subitems in iname.load_iter(coords=coords+[count]):
                        # print 'Yielding %s' % `subitems`
                        yield subitems
                    # print 'End of internal loop'
                else:
                    if self.dimension == 0:
                        # print 'Yielding %s' % `self[iname]`
                        yield self,self[iname]
                    else:
                        backval = self.block[iname]
                        for i in range(len(coords)):
                           # print 'backval, coords: %s, %s' % (`backval`,`coords`)
                           backval = backval[coords[i]]
                        yield self,backval
            count = count + 1      # count packets
        self.popout = False        # reinitialise
        # print 'Finished iterating'
        yield self,'###Blank###'     #this value should never be used

    # an experimental fast iterator for level-1 loops (ie CIF)
    def fast_load_iter(self):
        targets = map(lambda a:self.block[a],self.item_order)
        while targets:
            for target in targets:
                yield self,target

    # Add another list of the required shape to take into account a new outer packet
    def new_enclosing_packet(self):
        if self.dimension > 1:      #otherwise have a top-level list
            for iname in self.keys():  #includes lower levels
                target_list = self[iname]
                for i in range(3,self.dimension): #dim 2 upwards are lists of lists of... 
                    target_list = target_list[-1]
                target_list.append([])
                # print '%s now %s' % (iname,`self[iname]`)

    def recursive_iter(self,dict_so_far={},coord=[]):
        # print "Recursive iter: coord %s, keys %s, dim %d" % (`coord`,`self.block.keys()`,self.dimension)
        my_length = 0
        top_items = self.block.items()
        top_values = self.block.values()       #same order as items
        drill_values = self.block.values()
        for dimup in range(0,self.dimension):  #look higher in the tree
            if len(drill_values)>0:            #this block has values
                drill_values=drill_values[0]   #drill in
            else:
                raise StarError("Malformed loop packet %s" % `top_items[0]`)
        my_length = len(drill_values[0])       #length of 'string' entry
        if self.dimension == 0:                #top level
            for aloop in self.loops:
                for apacket in aloop.recursive_iter():
                    # print "Recursive yielding %s" % `dict(top_items + apacket.items())`
                    prep_yield = StarPacket(top_values+apacket.values())  #straight list
                    for name,value in top_items + apacket.items():
                        setattr(prep_yield,name,value)
                    yield prep_yield
        else:                                  #in some loop
            for i in range(my_length):
                kvpairs = map(lambda a:(a,self.coord_to_group(a,coord)[i]),self.block.keys())
                kvvals = map(lambda a:a[1],kvpairs)   #just values
                # print "Recursive kvpairs at %d: %s" % (i,`kvpairs`)
                if self.loops:
                  for aloop in self.loops:
                    for apacket in aloop.recursive_iter(coord=coord+[i]):
                        # print "Recursive yielding %s" % `dict(kvpairs + apacket.items())`
                        prep_yield = StarPacket(kvvals+apacket.values())
                        for name,value in kvpairs + apacket.items():
                            setattr(prep_yield,name,value)
                        yield prep_yield
                else:           # we're at the bottom of the tree
                    # print "Recursive yielding %s" % `dict(kvpairs)`
                    prep_yield = StarPacket(kvvals)
                    for name,value in kvpairs:
                        setattr(prep_yield,name,value)
                    yield prep_yield

    # small function to use the coordinates. 
    def coord_to_group(self,dataname,coords):
          if not isinstance(dataname,basestring):
             return dataname     # flag inner loop processing
          newm = self[dataname]          # newm must be a list or tuple
          for c in coords:
              # print "Coord_to_group: %s ->" % (`newm`),
              newm = newm[c]
              # print `newm`
          return newm 

    def flat_iterator(self):
            my_length = 0
            top_keys = self.block.keys()
            if len(top_keys)>0:
                my_length = len(self.block[top_keys[0]])
            for pack_no in range(my_length):
                yield(self.collapse(pack_no))
            

    def RemoveItem(self,itemname):
        """Remove `itemname` from the block."""
        # first check any loops
        loop_no = self.FindLoop(itemname)
        testkey = itemname.lower()
        if self.has_key(testkey):
            del self.block[testkey]
            del self.true_case[testkey]
            # now remove from loop
            if loop_no >= 0:
                self.loops[loop_no].remove(testkey)
                if len(self.loops[loop_no])==0:
                    del self.loops[loop_no]
                    self.item_order.remove(loop_no)
            else:  #will appear in order list
                self.item_order.remove(testkey)
     
    def RemoveLoopItem(self,itemname):
        """*Deprecated*. Use `RemoveItem` instead"""
        self.RemoveItem(itemname)

    def GetLoop(self,keyname):
        """Return a `StarFile.LoopBlock` object constructed from the loop containing `keyname`.
        `keyname` is only significant as a way to specify the loop."""
        return LoopBlock(self,keyname)

    def GetPacket(self,index):
        thispack = StarPacket([])
        for myitem in self.parent_block.loops[self.loop_no]:
            thispack.append(self[myitem][index])
            setattr(thispack,myitem,thispack[-1])
        return thispack 

    def AddPacket(self,packet):
        for myitem in self.parent_block.loops[self.loop_no]:
            old_values = self.parent_block[myitem]
            old_values.append(packet.__getattribute__(myitem))
            self.parent_block[myitem] = old_values
        
    def GetItemOrder(self):
        """Return a list of datanames in this `LoopBlock` in the order that they will be
        printed"""
        return self.parent_block.loops[self.loop_no][:]


    def GetItemOrder(self):
        """Return a list of datanames in this `LoopBlock` in the order that they will be
        printed"""
        return self.parent_block.loops[self.loop_no][:]

    def ChangeItemOrder(self,itemname,newpos):
        """Change the position at which `itemname` appears when printing out to `newpos`."""
        self.parent_block.loops[self.loop_no].remove(itemname.lower())
        self.parent_block.loops[self.loop_no].insert(newpos,itemname.lower())

    def GetItemPosition(self,itemname):
        """A utility function to get the numerical order in the printout
        of `itemname`.  An item has coordinate `(loop_no,pos)` with
        the top level having a `loop_no` of -1.  If an integer is passed to
        the routine then it will return the position of the loop
        referenced by that number."""
        import string
        if isinstance(itemname,int):
            # return loop position
            return (-1, self.item_order.index(itemname))
        if not self.has_key(itemname):
            raise ValueError, 'No such dataname %s' % itemname
        testname = itemname.lower()
        if testname in self.item_order:
            return (-1,self.item_order.index(testname))
        loop_no = self.FindLoop(testname)
        loop_pos = self.loops[loop_no].index(testname)
        return loop_no,loop_pos

    def GetLoopNames(self,keyname):
        if keyname in self:
            return self.keys()
        for aloop in self.loops:
            try: 
                return aloop.GetLoopNames(keyname)
            except KeyError:
                pass
        raise KeyError, 'Item does not exist'

    def GetLoopNames(self,keyname):
        """Return all datanames appearing together with `keyname`"""
        loop_no = self.FindLoop(keyname)
        if loop_no >= 0:
            return self.loops[loop_no]
        else:
            raise KeyError, '%s is not in any loop' % keyname

    def AddToLoop(self,dataname,loopdata):
        thisloop = self.GetLoop(dataname)
        for itemname,itemvalue in loopdata.items():
            thisloop[itemname] = itemvalue 
        
    def AddToLoop(self,dataname,loopdata):
        """*Deprecated*. Use `AddItem` followed by calls to `AddLoopName`.
        
        Add multiple columns to the loop containing `dataname`. `loopdata` is a
        collection of (key,value) pairs, where `key` is the new dataname and `value`
        is a list of values for that dataname"""
        # check lengths
        thisloop = self.FindLoop(dataname)
        loop_len = len(self[dataname])
        bad_vals = [a for a in loopdata.items() if len(a[1])!=loop_len]
        if len(bad_vals)>0:
           raise StarLengthError, "Number of values for looped datanames %s not equal to %d" \
               % (`bad_vals`,loop_len)
        self.update(loopdata)
        self.loops[thisloop]+=loopdata.keys()


class StarBlock(object):
    def __init__(self,data = (), maxoutlength=2048, wraplength=80, overwrite=True,
                 characterset='ascii',maxnamelength=-1):
        self.block = {}    #the actual data storage (lower case keys)
        self.loops = {}    #each loop is indexed by a number and contains a list of datanames
        self.item_order = []  #lower case, loops referenced by integer
        self.formatting_hints = {}
        self.true_case = {} #transform lower case to supplied case
        self.provide_value = False   #prefer string version always
        self.dictionary = None      #DDLm dictionary
        self.popout = False         #used during load iteration
        self.curitem = -1           #used during iteration
        self.maxoutlength = maxoutlength
        self.setmaxnamelength(maxnamelength)  #to enforce CIF limit of 75 characters
        self.set_characterset(characterset)   #to check input names
        self.wraplength = wraplength
        self.overwrite = overwrite
        self.string_delimiters = ["'",'"',"\n;"]   #universal CIF set
        self.list_delimiter = "  "                 #CIF2 default
        if isinstance(data,(tuple,list)):
            for item in data:
                self.AddLoopItem(item)
        elif isinstance(data,StarBlock):
            self.block = data.block.copy() 
            self.item_order = data.item_order[:]
            self.true_case = data.true_case.copy()
            # loops as well
            self.loops = data.loops.copy()

    def setmaxnamelength(self,maxlength):
        """Set the maximum allowable dataname length (-1 for no check)"""
        self.maxnamelength = maxlength
        if maxlength > 0:
            bad_names = [a for a in self.keys() if len(a)>self.maxnamelength]
            if len(bad_names)>0:
                raise StarError, 'Datanames too long: ' + `bad_names`

    def set_characterset(self,characterset):
        """Set the characterset for checking datanames: may be `ascii` or `unicode`"""
        import sys
        self.characterset = characterset
        if characterset == 'ascii':
            self.char_check = re.compile("[][ \n\r\t!%&\(\)*+,./:<=>?@0-9A-Za-z\\\\^`{}\|~\"#$';_-]+",re.M)
        elif characterset == 'unicode':
            if sys.maxunicode < 1114111:
               self.char_check = re.compile(u"[][ \n\r\t!%&\(\)*+,./:<=>?@0-9A-Za-z\\\\^`{}\|~\"#$';_\u00A0-\uD7FF\uE000-\uFDCF\uFDF0-\uFFFD-]+",re.M)
            else:
               self.char_check = re.compile(u"[][ \n\r\t!%&\(\)*+,./:<=>?@0-9A-Za-z\\\\^`{}\|~\"#$';_\u00A0-\uD7FF\uE000-\uFDCF\uFDF0-\uFFFD\U00010000-\U0010FFFD-]+",re.M)

    def __str__(self):
        return self.printsection()

    def __setitem__(self,key,value):
        if key == "saves":
            raise StarError("""Setting the saves key is deprecated. Add the save block to
    an enclosing block collection (e.g. CIF or STAR file) with this block as child""")
        self.AddItem(key,value)

    def __getitem__(self,key):
        if key == "saves":
            raise StarError("""The saves key is deprecated. Access the save block from
    the enclosing block collection (e.g. CIF or STAR file object)""") 
        try:
           rawitem,is_value = self.GetFullItemValue(key)
        except KeyError:
           if self.dictionary:
               # send the dictionary the required key and a pointer to us
               new_value = self.dictionary.derive_item(key,self,store_value=True)
               print 'Set %s to derived value %s' % (key, `new_value`)
               return new_value
           else:
               raise KeyError, 'No such item: %s' % key
        # we now have an item, we can try to convert it to a number if that is appropriate
        # note numpy values are never stored but are converted to lists
        if not self.dictionary or not self.dictionary.has_key(key): return rawitem
        print '%s: is_value %s provide_value %s value %s' % (key,`is_value`,`self.provide_value`,`rawitem`)
        if is_value:
            if self.provide_value: return rawitem
            else:
               print 'Turning %s into string' % `rawitem`
               return self.convert_to_string(key)
        else:    # a string
            if self.provide_value and ((not isinstance(rawitem,list) and rawitem != '?' and rawitem != ".") or \
                                      (isinstance(rawitem,list) and '?' not in rawitem and '.' not in rawitem)):
                return self.dictionary.change_type(key,rawitem)
            return rawitem   #can't do anything
           
    def __delitem__(self,key):
        self.RemoveItem(key)

    def __len__(self):
        blen = len(self.block)
        return blen    

    def __nonzero__(self):
        if self.__len__() > 0: return 1
        return 0

    # keys returns all internal keys
    def keys(self):
        return self.block.keys()    #always lower case

    def values(self):
        return [self[a] for a in self.keys()]

    def items(self):
        return [a for a in zip(self.keys(),self.values())]

    def has_key(self,key):
        if isinstance(key,basestring) and key.lower() in self.keys():
            return 1
        return 0

    def get(self,key,default=None):
        if self.has_key(key):
            retval = self.__getitem__(key)
        else:
            retval = default
        return retval

    def clear(self):
        self.block = {}
        self.loops = {}
        self.item_order = []
        self.true_case = {}

    # doesn't appear to work
    def copy(self):
        newcopy = StarBlock()
        newcopy.block = self.block.copy()
        newcopy.loops = []
        newcopy.item_order = self.item_order[:]
        newcopy.true_case = self.true_case.copy()
        newcopy.loops = self.loops.copy()
    #    return self.copy.im_class(newcopy)   #catch inheritance
        return newcopy
     
    def update(self,adict):
        for key in adict.keys():
            self.AddItem(key,adict[key])

    def GetItemPosition(self,itemname):
        """A utility function to get the numerical order in the printout
        of `itemname`.  An item has coordinate `(loop_no,pos)` with
        the top level having a `loop_no` of -1.  If an integer is passed to
        the routine then it will return the position of the loop
        referenced by that number."""
        import string
        if isinstance(itemname,int):
            # return loop position
            return (-1, self.item_order.index(itemname))
        if not self.has_key(itemname):
            raise ValueError, 'No such dataname %s' % itemname
        testname = itemname.lower()
        if testname in self.item_order:
            return (-1,self.item_order.index(testname))
        loop_no = self.FindLoop(testname)
        loop_pos = self.loops[loop_no].index(testname)
        return loop_no,loop_pos

    def ChangeItemOrder(self,itemname,newpos):
        """Move the printout order of `itemname` to `newpos`."""
        if isinstance(itemname,basestring):
            true_name = itemname.lower()
        else:
            true_name = itemname
        self.item_order.remove(true_name)
        self.item_order.insert(newpos,true_name)

    def GetItemOrder(self):
        """Return a list of datanames in the order in which they will be printed.  Loops are
        referred to by numerical index"""
        return self.item_order[:]

    def AddItem(self,key,value,precheck=False):
        """Add dataname `key` to block with value `value`.  `value` may be
        a single value, a list or a tuple."""
        if not isinstance(key,basestring):
             raise TypeError, 'Star datanames are strings only (got %s)' % `key`
        if not precheck:
             self.check_data_name(key,self.maxnamelength)    # make sure no nasty characters   
        # check for overwriting
        if self.has_key(key):
             if not self.overwrite:
                 raise StarError( 'Attempt to insert duplicate item name %s' % key)
        # put the data in the right container
        regval,empty_val = self.regularise_data(value)
        # check for pure string data
        pure_string = check_stringiness(regval)
        if not precheck:
            self.check_item_value(regval)
        # update ancillary information first
        lower_key = key.lower()
        if not self.has_key(lower_key) and self.FindLoop(lower_key)<0:      #need to add to order
            self.item_order.append(lower_key)
        # always remove from our case table in case the case is different
        else:
            del self.true_case[lower_key]
        self.true_case[lower_key] = key
        if pure_string:
            self.block.update({lower_key:[regval,empty_val]})
        else:
            self.block.update({lower_key:[empty_val,regval]})  
            
    def AddLoopItem(self,incomingdata,precheck=False,maxlength=-1):
        """*Deprecated*. Use `AddItem` followed by `CreateLoop` if
        necessary."""
        # print "Received data %s" % `incomingdata`
        # we accept tuples, strings, lists and dicts!!
        # Direct insertion: we have a string-valued key, with an array
        # of values -> single-item into our loop
        if isinstance(incomingdata[0],(tuple,list)):
           # a whole loop
           keyvallist = zip(incomingdata[0],incomingdata[1])
           for key,value in keyvallist:
               self.AddItem(key,value)
           self.CreateLoop(incomingdata[0])
        elif not isinstance(incomingdata[0],basestring):
             raise TypeError, 'Star datanames are strings only (got %s)' % `incomingdata[0]`
        else:
            self.AddItem(incomingdata[0],incomingdata[1])

    def check_data_name(self,dataname,maxlength=-1): 
        if maxlength > 0:
            self.check_name_length(dataname,maxlength)
        if dataname[0]!='_':
            raise StarError( 'Dataname ' + dataname + ' does not begin with _')
        if self.characterset=='ascii':
            if len (filter (lambda a: ord(a) < 33 or ord(a) > 126, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains forbidden characters')
        else:
            # print 'Checking %s for unicode characterset conformance' % dataname
            if len (filter (lambda a: ord(a) < 33, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains forbidden characters (below code point 33)')
            if len (filter (lambda a: ord(a) > 126 and ord(a) < 160, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains forbidden characters (between code point 127-159)')
            if len (filter (lambda a: ord(a) > 0xD7FF and ord(a) < 0xE000, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains unsupported characters (between U+D800 and U+E000)')
            if len (filter (lambda a: ord(a) > 0xFDCF and ord(a) < 0xFDF0, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains unsupported characters (between U+FDD0 and U+FDEF)')
            if len (filter (lambda a: ord(a) == 0xFFFE or ord(a) == 0xFFFF, dataname)) > 0:
                raise StarError( 'Dataname ' + dataname + ' contains unsupported characters (U+FFFE and/or U+FFFF)')
            if len (filter (lambda a: ord(a) > 0x10000 and (ord(a) & 0xE == 0xE) , dataname)) > 0:
                print '%s fails' % dataname
                for a in dataname: print '%x' % ord(a),
                print
                raise StarError( u'Dataname ' + dataname + u' contains unsupported characters (U+xFFFE and/or U+xFFFF)')

    def check_name_length(self,dataname,maxlength):
        if len(dataname)>maxlength:
            raise StarError( 'Dataname %s exceeds maximum length %d' % (dataname,maxlength))
        return

    def check_item_value(self,item):
        test_item = item
        if not isinstance(item,(list,dict,tuple)):
           test_item = [item]         #single item list
        def check_one (it):
            if isinstance(it,basestring):
                if it=='': return
                me = self.char_check.match(it)            
                if not me:
                    print "Fail value check: %s" % it
                    raise StarError, u'Bad character in %s' % it
                else:
                    if me.span() != (0,len(it)):
                        print "Fail value check, match only %d-%d in string %s" % (me.span()[0],me.span()[1],`it`)
                        raise StarError,u'Data item "' + `it` +  u'"... contains forbidden characters'
        map(check_one,test_item)

    def regularise_data(self,dataitem):
        """Place dataitem into a list if necessary"""
        from numbers import Number
        if isinstance(dataitem,(Number,basestring,StarList,StarDict)):
            return dataitem,None
        if isinstance(dataitem,(tuple,list)):
            return dataitem,[None]*len(dataitem)
        # so try to make into a list
        try:
            regval = list(dataitem)
        except TypeError, value:
            raise StarError( str(dataitem) + ' is wrong type for data value\n' )
        return regval,[None]*len(regval)
        
    def RemoveItem(self,itemname):
        """Remove `itemname` from the block."""
        # first check any loops
        loop_no = self.FindLoop(itemname)
        testkey = itemname.lower()
        if self.has_key(testkey):
            del self.block[testkey]
            del self.true_case[testkey]
            # now remove from loop
            if loop_no >= 0:
                self.loops[loop_no].remove(testkey)
                if len(self.loops[loop_no])==0:
                    del self.loops[loop_no]
                    self.item_order.remove(loop_no)
            else:  #will appear in order list
                self.item_order.remove(testkey)
     
    def RemoveLoopItem(self,itemname):
        """*Deprecated*. Use `RemoveItem` instead"""
        self.RemoveItem(itemname)

    def GetItemValue(self,itemname):
        """Return value of `itemname`.  If `itemname` is looped, a list
        of all values will be returned."""
        return self.GetFullItemValue(itemname)[0]

    def GetFullItemValue(self,itemname):
        """Return the value associated with `itemname`, and a boolean flagging whether 
        (True) or not (False) it is in a form suitable for calculation.  False is 
        always returned for strings and `StarList` objects."""
        try:
            s,v = self.block[itemname.lower()]
        except KeyError:
            raise KeyError, 'Itemname %s not in datablock' % itemname
        # prefer string value unless all are None
        # are we a looped value?
        if not isinstance(s,(tuple,list)) or isinstance(s,StarList):
            if s is not None or (isinstance(s,StarList) and None not in s): 
                return s,False    #a string value
            else:
                return v,not isinstance(v,StarList)  #a StarList is not calculation-ready
        elif None not in s: 
            return s,False     #a list of string values
        else: 
            if len(v)>0:   
                return v,not isinstance(v[0],StarList)
            return v,True

    def CreateLoop(self,datanames,order=-1): 
           """Create a loop in the datablock. `datanames` is a list of datanames that
           together form a loop.  They should have been initialised in the block
           to have the same number of elements (possibly 0). If `order` is given,
           the loop will appear at this position in the block when printing
           out. A loop counts as a single position."""

           # check lengths: these datanames should exist
           listed_values = [a for a in datanames if isinstance(self[a],list) and not isinstance(self[a],StarList)]
           if len(listed_values) == len(datanames):
               len_set = set([len(self[a]) for a in datanames])
               if len(len_set)>1:
                   raise ValueError, 'Request to loop datanames %s with different lengths: %s' % (`datanames`,`len_set`)
           elif len(listed_values) != 0:
               raise ValueError, 'Request to loop datanames where some are single values and some are not'
           # store as lower case
           lc_datanames = [d.lower() for d in datanames]
           # remove these datanames from all other loops
           [self.loops[a].remove(b) for a in self.loops for b in lc_datanames if b in self.loops[a]]
           # remove empty loops
           empty_loops = [a for a in self.loops.keys() if len(self.loops[a])==0]
           for a in empty_loops: del self.loops[a]
           if len(self.loops)>0:
               loopno = max(self.loops.keys()) + 1
           else:
               loopno = 1
           self.loops[loopno] = list(lc_datanames)
           if order >= 0:
               self.item_order.insert(order,loopno)
           else:
               self.item_order.append(loopno)
           # remove these datanames from item ordering
           self.item_order = [a for a in self.item_order if a not in lc_datanames] 
          
    def AddLoopName(self,oldname, newname):
        """Add `newname` to the loop containing `oldname`. If it is already in the new loop, no
        error is raised.  If `newname` is in a different loop, it is removed from that loop.
        The number of values associated with `newname` must match the number of values associated
        with all other columns of the new loop or a `ValueError` will be raised."""
        loop_no = self.FindLoop(oldname)
        if loop_no < 0:
            raise KeyError, '%s not in loop' % oldname
        if newname in self.loops[loop_no]:
            return
        # check length
        loop_len = len(self[oldname])
        if len(self[newname]) != loop_len:
            raise ValueError, 'Mismatch of loop column lengths for %s: should be %d' % (newname,loop_len)
        # remove from any other loops
        [self.loops[a].remove(newname) for a in self.loops if newname in self.loops[a]]
        # and add to this loop
        self.loops[loop_no].append(newname)
        
    def FindLoop(self,keyname):
        """Find the loop that contains `keyname` and return its numerical index or 
        -1 if not present. The numerical index can be used to refer to the loop in
        other routines."""
        loop_no = [a for a in self.loops.keys() if keyname.lower() in self.loops[a]]
        if len(loop_no)>0:
            return loop_no[0]
        else:
            return -1

    def GetLoop(self,keyname):
        """Return a `StarFile.LoopBlock` object constructed from the loop containing `keyname`.
        `keyname` is only significant as a way to specify the loop."""
        return LoopBlock(self,keyname)

    def GetLoopNames(self,keyname):
        if keyname in self:
            return self.keys()
        for aloop in self.loops:
            try: 
                return aloop.GetLoopNames(keyname)
            except KeyError:
                pass
        raise KeyError, 'Item does not exist'

    def GetLoopNames(self,keyname):
        """Return all datanames appearing together with `keyname`"""
        loop_no = self.FindLoop(keyname)
        if loop_no >= 0:
            return self.loops[loop_no]
        else:
            raise KeyError, '%s is not in any loop' % keyname

    def AddLoopName(self,oldname, newname):
        """Add `newname` to the loop containing `oldname`. If it is already in the new loop, no
        error is raised.  If `newname` is in a different loop, it is removed from that loop.
        The number of values associated with `newname` must match the number of values associated
        with all other columns of the new loop or a `ValueError` will be raised."""
        loop_no = self.FindLoop(oldname)
        if loop_no < 0:
            raise KeyError, '%s not in loop' % oldname
        if newname in self.loops[loop_no]:
            return
        # check length
        loop_len = len(self[oldname])
        if len(self[newname]) != loop_len:
            raise ValueError, 'Mismatch of loop column lengths for %s: should be %d' % (newname,loop_len)
        # remove from any other loops
        [self.loops[a].remove(newname) for a in self.loops if newname in self.loops[a]]
        # and add to this loop
        self.loops[loop_no].append(newname)
        
    def AddToLoop(self,dataname,loopdata):
        thisloop = self.GetLoop(dataname)
        for itemname,itemvalue in loopdata.items():
            thisloop[itemname] = itemvalue 
        
    def AddToLoop(self,dataname,loopdata):
        """*Deprecated*. Use `AddItem` followed by calls to `AddLoopName`.
        
        Add multiple columns to the loop containing `dataname`. `loopdata` is a
        collection of (key,value) pairs, where `key` is the new dataname and `value`
        is a list of values for that dataname"""
        # check lengths
        thisloop = self.FindLoop(dataname)
        loop_len = len(self[dataname])
        bad_vals = [a for a in loopdata.items() if len(a[1])!=loop_len]
        if len(bad_vals)>0:
           raise StarLengthError, "Number of values for looped datanames %s not equal to %d" \
               % (`bad_vals`,loop_len)
        self.update(loopdata)
        self.loops[thisloop]+=loopdata.keys()

    def RemoveKeyedPacket(self,keyname,keyvalue):
        """Remove the packet for which dataname `keyname` takes
        value `keyvalue`.  Only the first such occurrence is
        removed."""
        packet_coord = list(self[keyname]).index(keyvalue)
        loopnames = self.GetLoopNames(keyname)
        for dataname in loopnames:
            self.block[dataname][0] = list(self.block[dataname][0])
            del self.block[dataname][0][packet_coord]
            self.block[dataname][1] = list(self.block[dataname][1])
            del self.block[dataname][1][packet_coord]

    def GetKeyedPacket(self,keyname,keyvalue,no_case=False):
        """Return the loop packet (a `StarPacket` object) where `keyname` has value 
        `keyvalue`. Ignore case in `keyvalue` if `no_case` is True.  `ValueError`
        is raised if no packet is found or more than one packet is found."""
        #print "Looking for %s in %s" % (keyvalue, self.parent_block[keyname])
        my_loop = self.GetLoop(keyname)
        if no_case:
           one_pack= [a for a in my_loop if getattr(a,keyname).lower()==keyvalue.lower()]
        else:
           one_pack= [a for a in my_loop if getattr(a,keyname)==keyvalue]
        if len(one_pack)!=1:
            raise ValueError, "Bad packet key %s = %s: returned %d packets" % (keyname,keyvalue,len(one_pack))
        #print "Keyed packet: %s" % one_pack[0]
        return one_pack[0]

    def GetKeyedSemanticPacket(self,keyvalue,cat_id):
        """Return a complete packet for category `cat_id` where the
        category key for the category equals `keyvalue`.  This routine
        will understand any joined loops, so if separate loops in the
        datafile belong to the
        same category hierarchy (e.g. `_atom_site` and `_atom_site_aniso`),
        the returned `StarPacket` object will contain datanames from
        both categories."""
        target_keys = self.dictionary.cat_key_table[cat_id]
        p = StarPacket()
        # set case-sensitivity flag
        lcase = False
        if self.dictionary[target_keys[0]]['_type.contents'] in ['Code','Tag','Name']:
            lcase = True
        for cat_key in target_keys:
            try:
                extra_packet = self.GetKeyedPacket(cat_key,keyvalue,no_case=lcase)
            except KeyError:        #try to create the key
                key_vals = self[cat_key]   #will create a key column
            p.merge_packet(self.GetKeyedPacket(cat_key,keyvalue,no_case=lcase))
        # the following attributes used to calculate missing values
        p.key = target_keys[0]
        p.cif_dictionary = self.dictionary
        p.fulldata = self
        return p


    def set_grammar(self,new_grammar):
        self.string_delimiters = ["'",'"',"\n;",None]
        if new_grammar in ['STAR2','2.0']:
            self.string_delimiters += ['"""',"'''"]
        if new_grammar == '2.0':
            self.list_delimiter = "  "
        elif new_grammar == 'STAR2':
            self.list_delimiter = ", "
        else:
            self.list_delimiter = "*ERROR*"

    def SetOutputLength(self,wraplength=80,maxoutlength=2048):
        """Set the maximum output line length (`maxoutlength`) and the line length to
        wrap at (`wraplength`).  The wrap length is a target only and may not always be
        possible."""
        if wraplength > maxoutlength:
            raise StarError("Wrap length (requested %d) must be <= Maximum line length (requested %d)" % (wraplength,maxoutlength))
        self.wraplength = wraplength
        self.maxoutlength = maxoutlength

    def printsection(self,instring='',blockstart="",blockend="",indent=0,finish_at='',start_from=''):
        import string
        # first make an ordering
        self.create_ordering(finish_at,start_from)  #create self.output_order
        # now do it...
        if not instring:
            outstring = CIFStringIO(target_width=80)       # the returned string
        else:
            outstring = instring
        # print block delimiter
        outstring.write(blockstart,canbreak=True)
        while len(self.output_order)>0:
           #print "Remaining to output " + `self.output_order`
           itemname = self.output_order.pop(0)
           if not isinstance(itemname,int):  #no loop
                   item_spec = [i for i in self.formatting_hints if i['dataname'].lower()==itemname.lower()]
                   if len(item_spec)>0:
                       item_spec = item_spec[0]
                       col_pos = item_spec.get('column',-1)
                       name_pos = item_spec.get('name_pos',-1)
                   else:
                       col_pos = -1
                       item_spec = {}
                       name_pos = -1
                   if col_pos < 0: col_pos = 40
                   outstring.set_tab(col_pos)
                   itemvalue = self[itemname]
                   outstring.write(self.true_case[itemname],mustbreak=True,do_tab=False,startcol=name_pos)
                   outstring.write(' ',canbreak=True,do_tab=False,delimiter=True)    #space after itemname
                   self.format_value(itemvalue,outstring,hints=item_spec)
           else:# we are asked to print a loop block
                    outstring.set_tab(10)       #guess this is OK?
                    loop_spec = [i['name_pos'] for i in self.formatting_hints if i["dataname"]=='loop']
                    if loop_spec: 
                        loop_indent = max(loop_spec[0],0)
                    else:
                        loop_indent = indent
                    outstring.write('loop_\n',mustbreak=True,do_tab=False,startcol=loop_indent)
                    self.format_names(outstring,indent+2,loop_no=itemname)
                    self.format_packets(outstring,indent+2,loop_no=itemname)
        else:
            returnstring = outstring.getvalue()
        outstring.close()
        return returnstring

    def format_names(self,outstring,indent=0,loop_no=-1):
        """Print datanames from `loop_no` one per line"""
        temp_order = self.loops[loop_no][:]   #copy
        format_hints = dict([(i['dataname'],i) for i in self.formatting_hints if i['dataname'] in temp_order])
        while len(temp_order)>0:
            itemname = temp_order.pop(0)
            req_indent = format_hints.get(itemname,{}).get('name_pos',indent)
            outstring.write(' ' * req_indent,do_tab=False) 
            outstring.write(self.true_case[itemname],do_tab=False)
            outstring.write("\n",do_tab=False)

    def format_packets(self,outstring,indent=0,loop_no=-1):
       import cStringIO
       import string
       alldata = [self[a] for a in self.loops[loop_no]]
       loopnames = self.loops[loop_no]
       #print 'Alldata: %s' % `alldata`
       packet_data = apply(zip,alldata)
       #print 'Packet data: %s' % `packet_data`
       #create a dictionary for quick lookup of formatting requirements
       format_hints = dict([(i['dataname'],i) for i in self.formatting_hints if i['dataname'] in loopnames])
       for position in range(len(packet_data)):
           if position > 0:
               outstring.write("\n")    #new line each packet except first
           for point in range(len(packet_data[position])):
               datapoint = packet_data[position][point]
               format_hint = format_hints.get(loopnames[point],{})
               packstring = self.format_packet_item(datapoint,indent,outstring,format_hint)
               outstring.write(' ',canbreak=True,do_tab=False,delimiter=True)
               
    def format_packet_item(self,pack_item,indent,outstring,format_hint):
           # print 'Formatting %s' % `pack_item`
           if isinstance(pack_item,basestring):
               delimiter = format_hint.get('delimiter',None)
               startcol = format_hint.get('column',-1)
               outstring.write(self._formatstring(pack_item,delimiter=delimiter),startcol=startcol) 
           else: 
               self.format_value(pack_item,outstring,hints = format_hint)

    def _formatstring(self,instring,delimiter=None,standard='CIF1',indent=0):
        import string
        allowed_delimiters = set(self.string_delimiters)
        if len(instring)==0: allowed_delimiters.difference_update([None])
        if len(instring) > (self.maxoutlength-2) or '\n' in instring:
                allowed_delimiters.intersection_update(["\n;","'''",'"""'])
        if ' ' in instring or '\t' in instring or '\v' in instring or (len(instring)>0 and instring[0] in '_$#;([{') or ',' in instring:
                allowed_delimiters.difference_update([None])
        if '"' in instring: allowed_delimiters.difference_update(['"',None])
        if "'" in instring: allowed_delimiters.difference_update(["'",None])
        out_delimiter = "\n;"  #default (most conservative)
        if delimiter in allowed_delimiters:
            out_delimiter = delimiter
        elif "'" in allowed_delimiters: out_delimiter = "'"
        elif '"' in allowed_delimiters: out_delimiter = '"'
        if out_delimiter in ['"',"'",'"""',"'''"]: return out_delimiter + instring + out_delimiter
        elif out_delimiter is None: return instring
        # we are left with semicolon strings
        # use our protocols:
        maxlinelength = max([len(a) for a in instring.split('\n')])
        if maxlinelength > self.maxoutlength:
            protocol_string = apply_line_folding(instring)
        else:
            protocol_string = instring
        # now check for embedded delimiters
        if "\n;" in protocol_string:
            prefix = "CIF:"
            while prefix in protocol_string: prefix = prefix + ":"
            protocol_string = apply_line_prefix(protocol_string,prefix+"> ")
        return "\n;" + protocol_string + "\n;"

    def format_value(self,itemvalue,stringsink,compound=False,hints={}):
        """Format a Star data value"""
        delimiter = hints.get('delimiter',None)
        startcol = hints.get('column',-1)
        if isinstance(itemvalue,basestring):  #need to sanitize
           stringsink.write(self._formatstring(itemvalue,delimiter=delimiter),canbreak = True,startcol=startcol)
        elif isinstance(itemvalue,StarList):
           stringsink.set_tab(0)
           stringsink.write('[',canbreak=True,newindent=True,mustbreak=compound,startcol=startcol)
           if len(itemvalue)>0:
               self.format_value(itemvalue[0],stringsink)
               for listval in itemvalue[1:]:
                  # print 'Formatting %s' % `listval`
                  stringsink.write(self.list_delimiter,do_tab=False)
                  self.format_value(listval,stringsink,compound=True)
           stringsink.write(']',unindent=True)
        elif isinstance(itemvalue,StarDict):
           stringsink.set_tab(0)
           stringsink.write('{',newindent=True,mustbreak=compound,startcol=startcol)  #start a new line inside
           items = itemvalue.items()
           if len(items)>0:
               stringsink.write("'"+items[0][0]+"'"+':',canbreak=True)
               self.format_value(items[0][1],stringsink)
               for key,value in items[1:]:
                   stringsink.write(self.list_delimiter)
                   stringsink.write("'"+key+"'"+":",canbreak=True)
                   self.format_value(value,stringsink)   #never break between key and value
           stringsink.write('}',unindent=True)
        else: 
           stringsink.write(str(itemvalue),canbreak=True,startcol=startcol)   #numbers

    def create_ordering(self,finish_at,start_from):
        """Create a canonical ordering that includes loops using our formatting hints dictionary"""
        requested_order = [i['dataname'] for i in self.formatting_hints if i['dataname']!='loop']
        new_order = []
        for item in requested_order:
           if isinstance(item,basestring) and item.lower() in self.item_order:
               new_order.append(item.lower())
           elif self.has_key(item):    #in a loop somewhere
               target_loop = self.FindLoop(item)
               if target_loop not in new_order:
                   new_order.append(target_loop)
                   # adjust loop name order
                   loopnames = self.loops[target_loop]
                   loop_order = [i for i in requested_order if i in loopnames]
                   unordered = [i for i in loopnames if i not in loop_order]
                   self.loops[target_loop] = loop_order + unordered
        extras = [i for i in self.item_order if i not in new_order]
        self.output_order = new_order + extras
        # now handle partial output
        if start_from != '':
            loop_no = self.FindLoop(start_from)
            print '%s, loop_no is %d' % (start_from, loop_no)
            if self.output_order.count(start_from):
               self.output_order = self.output_order[self.output_order.index(start_from):]
            elif loop_no >= 0:
               self.output_order = self.output_order[self.output_order.index(loop_no):]
        if finish_at != '':
            loop_no = self.FindLoop(finish_at)
            if self.output_order.count(finish_at):
                self.output_order = self.output_order[:self.output_order.index(finish_at)]
            elif loop_no >= 0:
               self.output_order = self.output_order[:self.output_order.index(loop_no)]
        print 'Final order: ' + `self.output_order`

    def convert_to_string(self,dataname):
        """Convert values held in dataname value fork to string version"""
        v,is_value = self.GetFullItemValue(dataname)
        if not is_value:
            return v
        if check_stringiness(v): return v   #already strings
        # TODO...something else
        return v


    def merge(self,new_block,mode="strict",match_att=[],match_function=None,
                   rel_keys = []):
        if mode == 'strict':
           for key in new_block.keys(): 
               if self.has_key(key) and key not in match_att:
                  raise CifError( "Identical keys %s in strict merge mode" % key)
               elif key not in match_att:           #a new dataname
                   self[key] = new_block[key]
           # we get here if there are no keys in common, so we can now copy
           # the loops and not worry about overlaps
           for one_loop in new_block.loops.values():
               self.CreateLoop(one_loop)
           # we have lost case information
           self.true_case.update(new_block.true_case)
        elif mode == 'replace':
           newkeys = new_block.keys()
           for ma in match_att:
              try:
                   newkeys.remove(ma)        #don't touch the special ones
              except ValueError:
                   pass
           for key in new_block.keys(): 
                  if isinstance(key,basestring):
                      self[key] = new_block[key] 
           # creating the loop will remove items from other loops
           for one_loop in new_block.loops.values():
               self.CreateLoop(one_loop)
           # we have lost case information
           self.true_case.update(new_block.true_case)
        elif mode == 'overlay':
           print 'Overlay mode, current overwrite is %s' % self.overwrite
           raise StarError, 'Overlay block merge mode not implemented'
           save_overwrite = self.overwrite
           self.overwrite = True
           for attribute in new_block.keys():
               if attribute in match_att: continue      #ignore this one
               new_value = new_block[attribute]
               #non-looped items
               if new_block.FindLoop(attribute)<0:     #not looped
                  self[attribute] = new_value 
           my_loops = self.loops.values()
           perfect_overlaps = [a for a in new_block.loops if a in my_loops]
           for po in perfect_overlaps:
              loop_keys = [a for a in po if a in rel_keys]  #do we have a key?
              try:
                  newkeypos = map(lambda a:newkeys.index(a),loop_keys)
                  newkeypos = newkeypos[0]      #one key per loop for now
                  loop_keys = loop_keys[0] 
              except (ValueError,IndexError):
                  newkeypos = []
                  overlap_data = map(lambda a:listify(self[a]),overlaps) #old packet data
                  new_data = map(lambda a:new_block[a],overlaps) #new packet data
                  packet_data = transpose(overlap_data)
                  new_p_data = transpose(new_data)
                  # remove any packets for which the keys match between old and new; we
                  # make the arbitrary choice that the old data stays
                  if newkeypos:
                      # get matching values in new list
                      print "Old, new data:\n%s\n%s" % (`overlap_data[newkeypos]`,`new_data[newkeypos]`)
                      key_matches = filter(lambda a:a in overlap_data[newkeypos],new_data[newkeypos])
                      # filter out any new data with these key values
                      new_p_data = filter(lambda a:a[newkeypos] not in key_matches,new_p_data)
                      if new_p_data:
                          new_data = transpose(new_p_data)
                      else: new_data = []
                  # wipe out the old data and enter the new stuff
                  byebyeloop = self.GetLoop(overlaps[0])
                  # print "Removing '%s' with overlaps '%s'" % (`byebyeloop`,`overlaps`)
                  # Note that if, in the original dictionary, overlaps are not
                  # looped, GetLoop will return the block itself.  So we check
                  # for this case...
                  if byebyeloop != self:
                      self.remove_loop(byebyeloop)
                  self.AddLoopItem((overlaps,overlap_data))  #adding old packets
                  for pd in new_p_data:                             #adding new packets
                     if pd not in packet_data:
                        for i in range(len(overlaps)):
                            #don't do this at home; we are appending
                            #to something in place
                            self[overlaps[i]].append(pd[i]) 
           self.overwrite = save_overwrite

    def assign_dictionary(self,dic):
        if not dic.diclang=="DDLm":
            print "Warning: ignoring dictionary %s" % dic.dic_as_cif.my_uri
            return
        self.dictionary = dic

    def unassign_dictionary(self):
        """Remove dictionary-dependent behaviour"""
        self.dictionary = None



class StarPacket(list):
    def merge_packet(self,incoming):
        """Merge contents of incoming packet with this packet"""
        new_attrs = [a for a in dir(incoming) if a[0] == '_' and a[1] != "_"]
        self.append(incoming)
        for na in new_attrs:
            setattr(self,na,getattr(incoming,na))

    def __getattr__(self,att_name):
        """Derive a missing attribute"""
        if att_name.lower() in self.__dict__:
            return getattr(self,att_name.lower())
        if att_name in ('cif_dictionary','fulldata','key'):
            raise AttributeError, 'Programming error: cannot compute value of %s' % att_name
        d = self.cif_dictionary
        c = self.fulldata
        k = self.key
        d.derive_item(att_name,c,store_value=True)
        # 
        # now pick out the new value
        keyval = getattr(self,k)
        full_pack = c.GetKeyedPacket(k,keyval)
        return getattr(full_pack,att_name)
        
class BlockCollection(object):
    """A container for StarBlock objects. The constructor takes
    one non-keyword argument `datasource` to set the initial data.  If
    `datasource` is a Python dictionary, the values must be `StarBlock`
    objects and the keys will be blocknames in the new object. Keyword
    arguments: 

    standard: 
        `CIF` or `Dic`.  `CIF` enforces 75-character blocknames, and will
        print block contents before that block's save frame.
  
    blocktype:
        The type of blocks held in this container. Normally `StarBlock`
        or `CifBlock`.

    characterset:
        `ascii` or `unicode`.  Blocknames and datanames appearing within
        blocks are restricted to the appropriate characterset. Note that
        only characters in the basic multilingual plane are accepted. This
        restriction will be lifted when PyCIFRW is ported to Python3.

    scoping:
        `instance` or `dictionary`: `instance` implies that save frames are
        hidden from save frames lower in the hierarchy or in sibling
        hierarchies. `dictionary` makes all save frames visible everywhere
        within a data block.  This setting is only relevant for STAR2 dictionaries and
        STAR2 data files, as save frames are currently not used in plain CIF data
        files.

"""
    def __init__(self,datasource=None,standard='CIF',blocktype = StarBlock,
                 characterset='ascii',scoping='instance',**kwargs):
        import collections
        self.dictionary = {}
        self.standard = standard
        self.lower_keys = set()           # short_cuts
        self.renamed = {}
        self.PC = collections.namedtuple('PC',['block_id','parent'])
        self.child_table = {}
        self.visible_keys = []            # for efficiency
        self.scoping = scoping  #will trigger setting of child table
        self.blocktype = blocktype
        self.master_template = {}   #for outputting
        self.set_grammar('2.0')
        self.set_characterset(characterset)
        if isinstance(datasource,BlockCollection):
            self.merge_fast(datasource)
            self.scoping = scoping   #reset visibility
        elif isinstance(datasource,dict):
            for key,value in datasource.items():
                 self[key]= value
        self.header_comment = ''
     
    def set_grammar(self,new_grammar):
        """Set the syntax and grammar for output to `new_grammar`"""
        self.grammar = new_grammar

    def set_characterset(self,characterset):
        """Set the allowed characters for datanames and datablocks: may be `ascii` or `unicode`. If datanames
        have already been added to any datablocks, they are not checked."""
        self.characterset = characterset
        for one_block in self.lower_keys:
            self[one_block].set_characterset(characterset)

    def unlock(self):
        """Allow overwriting of all blocks in this collection"""
        for a in self.lower_keys:
            self[a].overwrite=True

    def lock(self):
        """Disallow overwriting for all blocks in this collection"""
        for a in self.lower_keys:
            self[a].overwrite = False

    def __str__(self):
        return self.WriteOut()

    def __setitem__(self,key,value):
        self.NewBlock(key,value,parent=None)

    def __getitem__(self,key):
        if isinstance(key,basestring):
           lowerkey = key.lower()
           if lowerkey in self.lower_keys:
               return self.dictionary[lowerkey]
           #print 'Visible keys:' + `self.visible_keys`
           #print 'All keys' + `self.lower_keys`
           #print 'Child table' + `self.child_table`
           raise KeyError,'No such item %s' % key

    # we have to get an ordered list of the current keys,
    # as we'll have to delete one of them anyway.
    # Deletion will delete any key regardless of visibility

    def __delitem__(self,key):
        dummy = self[key]   #raise error if not present
        lowerkey = key.lower()
        # get rid of all children recursively as well
        children = [a[0] for a in self.child_table.items() if a[1].parent == lowerkey]
        for child in children:
            del self[child]   #recursive call
        del self.dictionary[lowerkey]
        del self.child_table[lowerkey]
        try:
            self.visible_keys.remove(lowerkey)
        except KeyError:
            pass
        self.lower_keys.remove(lowerkey)

    def __len__(self):
        return len(self.visible_keys)

    def __contains__(self,item):
        """Support the 'in' operator"""
        return self.has_key(item)

    # We iterate over all visible
    def __iter__(self):
        for one_block in self.keys():
            yield self[one_block]

    # TODO: handle different case
    def keys(self):
        return self.visible_keys

    # changes to take case independence into account
    def has_key(self,key):
        if not isinstance(key,basestring): return 0
        if key.lower() in self.visible_keys:
            return 1
        return 0

    def get(self,key,default=None):
        if self.has_key(key):     # take account of case
            return self.__getitem__(key)
        else:
            return default

    def clear(self):
        self.dictionary.clear()
        self.lower_keys = set()
        self.child_table = {}
        self.visible_keys = []

    def copy(self):   
        newcopy = self.dictionary.copy()  #all blocks
        for k,v in self.dictionary.items():
            newcopy[k] = v.copy()
        newcopy = BlockCollection(newcopy)
        newcopy.child_table = self.child_table.copy()
        newcopy.lower_keys = self.lower_keys.copy()
        newcopy.characterset = self.characterset
        newcopy.master_template = self.master_template.copy()
        newcopy.scoping = self.scoping  #this sets visible keys
        return newcopy

    def update(self,adict):
        for key in adict.keys():
            self[key] = adict[key]

    def items(self):
        return [(a,self[a]) for a in self.keys()]

    def first_block(self):
        """Return the 'first' block.  This is not necessarily the first block in the file."""
        if self.keys():
            return self[self.keys()[0]]

    def NewBlock(self,blockname,blockcontents=None,fix=True,parent=None):
        """Add a new block named `blockname` with contents `blockcontents`. If `fix`
        is True, `blockname` will have spaces and tabs replaced by underscores. `parent`
        allows a parent block to be set so that block hierarchies can be created.  Depending on
        the output standard, these blocks will be printed out as nested save frames or
        ignored."""
        if blockcontents is None:
            blockcontents = StarBlock()
        if self.standard == "CIF":
            blockcontents.setmaxnamelength(75)
        if len(blockname)>75:
                 raise StarError , 'Blockname %s is longer than 75 characters' % blockname
        if fix:
            newblockname = re.sub('[  \t]','_',blockname)
        else: newblockname = blockname
        new_lowerbn = newblockname.lower()
        if new_lowerbn in self.lower_keys:
            if self.standard is not None:    #already there
               toplevelnames = [a[0] for a in self.child_table.items() if a[1].parent==None]
               if parent is None and new_lowerbn not in toplevelnames:  #can give a new key to this one
                  while new_lowerbn in self.lower_keys: new_lowerbn = new_lowerbn + '+'
               elif parent is not None and new_lowerbn in toplevelnames: #can fix a different one
                  replace_name = new_lowerbn            
                  while replace_name in self.lower_keys: replace_name = replace_name + '+'
                  self._rekey(new_lowerbn,replace_name)
                  # now continue on to add in the new block
                  if parent.lower() == new_lowerbn:        #the new block's requested parent just got renamed!!
                      parent = replace_name
               else:
                  raise StarError( "Attempt to replace existing block " + blockname)
            else:
               del self[new_lowerbn]
        self.dictionary.update({new_lowerbn:blockcontents})
        self.lower_keys.add(new_lowerbn)
        if parent is None:
           self.child_table[new_lowerbn]=self.PC(newblockname,None)
           self.visible_keys.append(new_lowerbn)
        else:
           if parent.lower() in self.lower_keys:
              if self.scoping == 'instance':
                 self.child_table[new_lowerbn]=self.PC(newblockname,parent.lower())
              else:
                 self.child_table[new_lowerbn]=self.PC(newblockname,parent.lower())
                 self.visible_keys.append(new_lowerbn)
           else:
               print 'Warning:Parent block %s does not exist for child %s' % (parent,newblockname)  
        self[new_lowerbn].set_grammar(self.grammar)
        self[new_lowerbn].set_characterset(self.characterset)
        return new_lowerbn  #in case calling routine wants to know

    def _rekey(self,oldname,newname,block_id=''):
        """The block with key [[oldname]] gets [[newname]] as a new key, but the printed name
           does not change unless [[block_id]] is given.  Prefer [[rename]] for a safe version."""
        move_block = self[oldname]    #old block
        is_visible = oldname in self.visible_keys
        move_block_info = self.child_table[oldname]    #old info
        move_block_children = [a for a in self.child_table.items() if a[1].parent==oldname]
        # now rewrite the necessary bits
        self.child_table.update(dict([(a[0],self.PC(a[1].block_id,newname)) for a in move_block_children]))
        del self[oldname]   #do this after updating child table so we don't delete children
        self.dictionary.update({newname:move_block})
        self.lower_keys.add(newname)
        if block_id == '':
           self.child_table.update({newname:move_block_info})
        else:
           self.child_table.update({newname:self.PC(block_id,move_block_info.parent)})
        if is_visible: self.visible_keys += [newname]

    def rename(self,oldname,newname):
        """Rename datablock from [[oldname]] to [[newname]]. Both key and printed name are changed.  No
           conformance checks are conducted."""
        realoldname = oldname.lower()
        realnewname = newname.lower()
        if realnewname in self.lower_keys:
            raise StarError,'Cannot change blockname %s to %s as %s already present' % (oldname,newname,newname)
        if realoldname not in self.lower_keys:
            raise KeyError,'Cannot find old block %s' % realoldname
        self._rekey(realoldname,realnewname,block_id=newname)
        
    def merge_fast(self,new_bc,parent=None):
        """Do a fast merge"""
        if self.standard is None:
            mode = 'replace' 
        else:
            mode = 'strict'
        overlap_flag = not self.lower_keys.isdisjoint(new_bc.lower_keys)
        if overlap_flag and mode != 'replace':
            double_keys = self.lower_keys.intersection(new_bc.lower_keys)
            for dup_key in double_keys:
                  our_parent = self.child_table[dup_key].parent
                  their_parent = new_bc.child_table[dup_key].parent
                  if (our_parent is None and their_parent is not None and parent is None) or\
                      parent is not None:  #rename our block
                    start_key = dup_key
                    while start_key in self.lower_keys: start_key = start_key+'+'
                    self._rekey(dup_key,start_key)
                    if parent.lower() == dup_key:  #we just renamed the prospective parent!
                        parent = start_key
                  elif our_parent is not None and their_parent is None and parent is None:
                    start_key = dup_key
                    while start_key in new_bc.lower_keys: start_key = start_key+'+'
                    new_bc._rekey(dup_key,start_key)
                  else: 
                    raise StarError("In strict merge mode:duplicate keys %s" % dup_key)
        self.dictionary.update(new_bc.dictionary) 
        self.lower_keys.update(new_bc.lower_keys)
        self.visible_keys += (list(new_bc.lower_keys))
        self.child_table.update(new_bc.child_table)
        if parent is not None:     #redo the child_table entries
              reparent_list = [(a[0],a[1].block_id) for a in new_bc.child_table.items() if a[1].parent==None]
              reparent_dict = [(a[0],self.PC(a[1],parent.lower())) for a in reparent_list]
              self.child_table.update(dict(reparent_dict))

    def merge(self,new_bc,mode=None,parent=None,single_block=[],
                   idblock="",match_att=[],match_function=None):
        if mode is None:
            if self.standard is None:
               mode = 'replace'
            else:
               mode = 'strict'
        if single_block:
            self[single_block[0]].merge(new_bc[single_block[1]],mode,
                                                   match_att=match_att,
                                                   match_function=match_function)
            return None
        base_keys = [a[1].block_id for a in self.child_table.items()]
        block_to_item = base_keys   #default
        new_keys = [a[1].block_id for a in new_bc.child_table.items()]    #get list of incoming blocks
        if match_att:
            #make a blockname -> item name map
            if match_function:
                block_to_item = map(lambda a:match_function(self[a]),self.keys())
            else:
                block_to_item = map(lambda a:self[a].get(match_att[0],None),self.keys())
            #print `block_to_item`
        for key in new_keys:        #run over incoming blocknames
            if key == idblock: continue    #skip dictionary id
            basekey = key           #default value
            if len(match_att)>0:
               attval = new_bc[key].get(match_att[0],0)  #0 if ignoring matching
            else:
               attval = 0
            for ii in range(len(block_to_item)):  #do this way to get looped names
                thisatt = block_to_item[ii]       #keyname in old block
                #print "Looking for %s in %s" % (attval,thisatt)
                if attval == thisatt or \
                   (isinstance(thisatt,list) and attval in thisatt):
                      basekey = base_keys.pop(ii)
                      block_to_item.remove(thisatt)
                      break
            if not self.has_key(basekey) or mode=="replace":
                new_parent = new_bc.get_parent(key)
                if parent is not None and new_parent is None:
                   new_parent = parent
                self.NewBlock(basekey,new_bc[key],parent=new_parent)   #add the block
            else:
                if mode=="strict":
                    raise StarError( "In strict merge mode: block %s in old and block %s in new files" % (basekey,key))
                elif mode=="overlay":
                    # print "Merging block %s with %s" % (basekey,key)
                    self[basekey].merge(new_bc[key],mode,match_att=match_att)
                else:  
                    raise StarError( "Merge called with unknown mode %s" % mode)
         
    def checknamelengths(self,target_block,maxlength=-1):
        if maxlength < 0:
            return
        else:
            toolong = filter(lambda a:len(a)>maxlength, target_block.keys())
        outstring = ""
        for it in toolong: outstring += "\n" + it
        if toolong:
           raise StarError( 'Following data names too long:' + outstring)

    def get_all(self,item_name):
        raw_values = map(lambda a:self[a].get(item_name),self.keys())
        raw_values = filter(lambda a:a != None, raw_values)
        ret_vals = []
        for rv in raw_values:
            if isinstance(rv,list):
                for rvv in rv:
                    if rvv not in ret_vals: ret_vals.append(rvv)
            else:
                if rv not in ret_vals: ret_vals.append(rv)
        return ret_vals

    def __setattr__(self,attr_name,newval):
        if attr_name == 'scoping':
            if newval not in ('dictionary','instance'):
                raise StarError("Star file may only have 'dictionary' or 'instance' scoping, not %s" % newval)
            if newval == 'dictionary':
                self.visible_keys = [a for a in self.lower_keys] 
            else:
                #only top-level datablocks visible
                self.visible_keys = [a[0] for a in self.child_table.items() if a[1].parent==None]
        object.__setattr__(self,attr_name,newval)

    def get_parent(self,blockname):
        """Return the name of the block enclosing [[blockname]] in canonical form (lower case)"""
        possibles = (a for a in self.child_table.items() if a[0] == blockname.lower())
        try:
            first = possibles.next()   #get first one
        except:
            raise StarError('no parent for %s' % blockname)
        try:
           second = possibles.next()
        except StopIteration:
           return first[1].parent
        raise StarError('More than one parent for %s' % blockname)

    def get_roots(self):
        """Get the top-level blocks"""
        return [a for a in self.child_table.items() if a[1].parent==None]

    def get_children(self,blockname,include_parent=False,scoping='dictionary'):
        """Get all children of [[blockname]] as a block collection. If [[include_parent]] is
        True, the parent block will also be included in the block collection as the root."""
        newbc = BlockCollection()
        block_lower = blockname.lower()
        proto_child_table = [a for a in self.child_table.items() if self.is_child_of_parent(block_lower,a[1].block_id)]
        newbc.child_table = dict(proto_child_table)
        if not include_parent:
           newbc.child_table.update(dict([(a[0],self.PC(a[1].block_id,None)) for a in proto_child_table if a[1].parent == block_lower]))
        newbc.lower_keys = set([a[0] for a in proto_child_table])
        newbc.dictionary = dict((a[0],self.dictionary[a[0]]) for a in proto_child_table)
        if include_parent:
            newbc.child_table.update({block_lower:self.PC(self.child_table[block_lower].block_id,None)})
            newbc.lower_keys.add(block_lower)
            newbc.dictionary.update({block_lower:self.dictionary[block_lower]})
        newbc.scoping = scoping
        return newbc

    def get_immediate_children(self,parentname):
        """Get the next level of children of the given block as a list, without nested levels"""
        child_handles = [a for a in self.child_table.items() if a[1].parent == parentname.lower()]
        return child_handles

    # This takes time
    def get_child_list(self,parentname):
        """Get a list of all child categories in alphabetical order"""
        child_handles = [a[0] for a in self.child_table.items() if self.is_child_of_parent(parentname.lower(),a[0])]
        child_handles.sort()
        return child_handles

    def is_child_of_parent(self,parentname,blockname):
        """Return `True` if `blockname` is a child of `parentname`"""
        checkname = parentname.lower()
        more_children = [a[0] for a in self.child_table.items() if a[1].parent == checkname]
        if blockname.lower() in more_children:
           return True
        else:
           for one_child in more_children:
               if self.is_child_of_parent(one_child,blockname): return True
        return False
           
    def set_parent(self,parentname,childname):
        """Set the parent block"""
        # first check that both blocks exist
        if parentname.lower() not in self.lower_keys:
            raise KeyError('Parent block %s does not exist' % parentname)
        if childname.lower() not in self.lower_keys:
            raise KeyError('Child block %s does not exist' % childname)
        old_entry = self.child_table[childname.lower()]
        self.child_table[childname.lower()]=self.PC(old_entry.block_id,
               parentname.lower())
        self.scoping = self.scoping #reset visibility
            
    def SetTemplate(self,template_file):
            """Use `template_file` as a template for all block output"""
            self.master_template = process_template(template_file)

    def WriteOut(self,comment='',wraplength=80,maxoutlength=0,blockorder=None,saves_after=None):
        """Return the contents of this file as a string, wrapping if possible at `wraplength`
        characters and restricting maximum line length to `maxoutlength`.  Delimiters and
        save frame nesting are controlled by `self.grammar`. If `blockorder` is
        provided, blocks are output in this order unless nested save frames have been
        requested (STAR2). `saves_after` inserts all save frames after the given dataname,
        which allows less important items to appear later.  Useful in conjunction with a
        template for dictionary files."""
        import cStringIO
        if maxoutlength != 0:
            self.SetOutputLength(maxoutlength)
        if not comment:
            comment = self.header_comment
        outstring = cStringIO.StringIO()
        if self.grammar == "2.0" and comment[0:10] != r"#\#CIF_2.0":
            outstring.write(r"#\#CIF_2.0" + "\n")
        outstring.write(comment)
        # prepare all blocks
        for b in self.dictionary.values():
            b.set_grammar(self.grammar)
            b.formatting_hints = self.master_template
            b.SetOutputLength(wraplength,self.maxoutlength)
        # loop over top-level
        # monitor output
        all_names = self.child_table.keys()
        if blockorder is None:
            top_block_names = [(a[0],a[1].block_id) for a in self.child_table.items() if a[1].parent is None]
        else:
            top_block_names = [(a,self.child_table[a].block_id) for a in blockorder if self.child_table[a].parent is None]
        for blockref,blockname in top_block_names:
            print 'Writing %s, ' % blockname + `self[blockref]`
            outstring.write('\n' + 'data_' +blockname+'\n')
            all_names.remove(blockref)
            if self.standard == 'Dic':              #put contents before save frames
                outstring.write(self[blockref].printsection(finish_at='_dictionary_valid.application'))
            if self.grammar == 'STAR2':  #nested save frames
                child_refs = self.get_immediate_children(blockref)
                for child_ref,child_info in child_refs:
                    child_name = child_info.block_id
                    outstring.write('\n\n' + 'save_' + child_name + '\n')
                    self.block_to_string_nested(child_ref,child_name,outstring,4)    
                    outstring.write('\n' + 'save_'+ '\n')   
            elif self.grammar in ('1.0','1.1','2.0'):                   #non-nested save frames
                if blockorder is None:
                    child_refs = self.get_child_list(blockref)
                else:
                    child_refs = [a for a in blockorder if self.is_child_of_parent(blockref,a)]
                for child_ref in child_refs:
                    child_name = self.child_table[child_ref].block_id
                    outstring.write('\n\n' + 'save_' + child_name + '\n')
                    outstring.write(str(self[child_ref]))
                    outstring.write('\n\n' + 'save_' + '\n')
                    all_names.remove(child_ref)
            if self.standard != 'Dic':              #put contents after save frames
                outstring.write(str(self[blockref]))
            else:
                outstring.write(self[blockref].printsection(start_from='_dictionary_valid.application'))
        returnstring =  outstring.getvalue()
        outstring.close()
        if len(all_names)>0:
            print 'WARNING: following blocks not output: %s' % `all_names`
        return returnstring

    def block_to_string_nested(self,block_ref,block_id,outstring,indentlevel=0):
        """Output a complete datablock indexed by [[block_ref]] and named [[block_id]], including children,
           and syntactically nesting save frames"""
        child_refs = self.get_immediate_children(block_ref)
        self[block_ref].set_grammar(self.grammar)
        if self.standard == 'Dic':
            outstring.write(str(self[block_ref]))
        for child_ref,child_info in child_refs:
            child_name = child_info.block_id
            outstring.write('\n' + 'save_' + child_name + '\n')
            self.block_to_string_nested(child_ref,child_name,outstring,indentlevel)
            outstring.write('\n' + '  '*indentlevel + 'save_' + '\n')
        if self.standard != 'Dic':
            outstring.write(str(self[block_ref]))


class StarFile(BlockCollection):
    def __init__(self,datasource=None,maxinlength=-1,maxoutlength=0,
                scoping='instance',grammar='1.1',scantype='standard',
                **kwargs):
        super(StarFile,self).__init__(datasource=datasource,**kwargs)
        self.my_uri = getattr(datasource,'my_uri','')
        if maxoutlength == 0:
            self.maxoutlength = 2048 
        else:
            self.maxoutlength = maxoutlength
        self.scoping = scoping
        if isinstance(datasource,basestring) or hasattr(datasource,"read"):
            ReadStar(datasource,prepared=self,grammar=grammar,scantype=scantype,
            maxlength = maxinlength)
        self.header_comment = \
"""#\\#STAR
##########################################################################
#               STAR Format file 
#               Produced by PySTARRW module
# 
#  This is a STAR file.  STAR is a superset of the CIF file type.  For
#  more information, please refer to International Tables for Crystallography,
#  Volume G, Chapter 2.1
#
##########################################################################
"""
    def set_uri(self,my_uri): self.my_uri = my_uri


from StringIO import StringIO
import math
class CIFStringIO(StringIO):
    def __init__(self,target_width=80,**kwargs):
        StringIO.__init__(self,**kwargs)
        self.currentpos = 0
        self.target_width = target_width
        self.tabwidth = -1
        self.indentlist = [0]
        self.last_char = ""

    def write(self,outstring,canbreak=False,mustbreak=False,do_tab=True,newindent=False,unindent=False,
                             delimiter=False,startcol=-1):
        """Write a string with correct linebreak, tabs and indents"""
        # do we need to break?
        if delimiter:
            if len(outstring)>1:
                raise ValueError, 'Delimiter %s is longer than one character' % `outstring`
            output_delimiter = True
        if mustbreak:    #insert a new line and indent
            temp_string = '\n' + ' ' * self.indentlist[-1]
            StringIO.write(self,temp_string)
            self.currentpos = self.indentlist[-1]
            self.last_char = temp_string[-1]
        if self.currentpos+len(outstring)>self.target_width: #try to break
            if not delimiter and outstring[0]!='\n':          #ie <cr>;  
              if canbreak:
                temp_string = '\n' + ' ' * self.indentlist[-1]
                StringIO.write(self,temp_string)
                self.currentpos = self.indentlist[-1]
                self.last_char = temp_string[-1]
            else:        #assume a break will be forced on next value
                output_delimiter = False    #the line break becomes the delimiter
        #try to match requested column
        if startcol > 0:
            if self.currentpos < startcol:
                StringIO.write(self,(startcol - self.currentpos)* ' ')
                self.currentpos = startcol
                self.last_char = ' '
            else:
                print 'Could not format %s at column %d as already at %d' % (outstring,startcol,self.currentpos)
                startcol = -1   #so that tabbing works as a backup
        #handle tabs
        if self.tabwidth >0 and do_tab and startcol < 0:
            next_stop = ((self.currentpos//self.tabwidth)+1)*self.tabwidth
            #print 'Currentpos %d: Next tab stop at %d' % (self.currentpos,next_stop)
            if self.currentpos < next_stop:
                StringIO.write(self,(next_stop-self.currentpos)*' ')
                self.currentpos = next_stop
                self.last_char = ' '
        #calculate indentation after tabs and col setting applied
        if newindent:           #indent by current amount
            if self.indentlist[-1] == 0:    #first time
                self.indentlist.append(self.currentpos)
                # print 'Indentlist: ' + `self.indentlist`
            else:
                self.indentlist.append(self.indentlist[-1]+2)
        elif unindent:
            if len(self.indentlist)>1:
                self.indentlist.pop()
            else:
                print 'Warning: cannot unindent any further'
        #check that we still need a delimiter
        if self.last_char in [' ','\n','\t']:
            output_delimiter = False
        #now output the string - every invocation comes through here
        if (delimiter and output_delimiter) or not delimiter:
            StringIO.write(self,outstring)
        last_line_break = outstring.rfind('\n')
        if last_line_break >=0:
            self.currentpos = len(outstring)-last_line_break
        else:
            self.currentpos = self.currentpos + len(outstring)
        #remember the last character
        if len(outstring)>0:
            self.last_char = outstring[-1]
        
    def set_tab(self,tabwidth):
        """Set the tab stop position"""
        self.tabwidth = tabwidth

class StarError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return '\nStar Format error: '+ self.value 

class StarLengthError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return '\nStar length error: ' + self.value

def ReadStar(filename,prepared = None, maxlength=-1,
             scantype='standard',grammar='STAR2',CBF=False):

    """ Read in a STAR file, returning the contents in the `prepared` object.  

    * `filename` may be a URL, a file
    path on the local system, or any object with a `read` method.

    * `prepared` provides a `StarFile` or `CifFile` object that the contents of `filename`
    will be added to.
    
    * `maxlength` is the maximum allowable line length in the input file. This has been set at
    2048 characters for CIF but is unlimited (-1) for STAR files.

    * `grammar` chooses the STAR grammar variant. `1.0` is the original 1992 CIF/STAR grammar and `1.1`
    is identical except for the exclusion of square brackets as the first characters in
    undelimited datanames. `2.0` will read files in the CIF2.0 standard, and `STAR2` will
    read files according to the STAR2 publication.  If grammar is `None` or `auto`, autodetection
    will be attempted in the order `2.0`, `1.1` and `1.0`. This will always succeed for conformant CIF2.0 files.
    Note that (nested) save frames are read in all grammar variations and then flagged afterwards if
    they do not match the requested grammar.

    * `scantype` can be `standard` or `flex`.  `standard` provides pure Python parsing at the
    cost of a factor of 10 or so in speed.  `flex` will tokenise the input CIF file using
    fast C routines.  Note that running PyCIFRW in Jython uses native Java regular expressions
    to provide a speedup regardless of this argument.

    * `CBF` flags that the input file is in Crystallographic Binary File format. The binary block is
    excised from the input data stream before parsing and is not available in the returned object.
    """

    import string
    import codecs
    # save desired scoping
    save_scoping = prepared.scoping
    import YappsStarParser_1_1 as Y11
    import YappsStarParser_1_0 as Y10
    import YappsStarParser_2_0 as Y20
    import YappsStarParser_STAR2 as YST
    if prepared is None:
        prepared = StarFile()
    if grammar == "auto" or grammar is None:
        try_list = [('2.0',Y20),('1.1',Y11),('1.0',Y10)]
    elif grammar == '1.0':
        try_list = [('1.0',Y10)]
    elif grammar == '1.1':
        try_list = [('1.1',Y11)]
    elif grammar == '2.0':
        try_list = [('2.0',Y20)]
    elif grammar == 'STAR2':
        try_list = [('STAR2',YST)]
    else:
        raise AttributeError, 'Unknown STAR/CIF grammar requested, %s' % `grammar`
    if isinstance(filename,basestring):
        filestream = urlopen(filename)
    else:
        filestream = filename   #already opened for us
    my_uri = ""
    if hasattr(filestream,"geturl"): 
        my_uri = filestream.geturl()
    text = unicode(filestream.read(),"utf8")
    if isinstance(filename,basestring): #we opened it, we close it
        filestream.close()
    if not text:      # empty file, return empty block
        return prepared.set_uri(my_uri)
    # filter out non-ASCII characters in CBF files if required.  We assume
    # that the binary is enclosed in a fixed string that occurs
    # nowhere else.
    if CBF:
       text_bits  = text.split("-BINARY-FORMAT-SECTION-") 
       text = text_bits[0] 
       for section in range(2,len(text_bits),2):
           text = text+" (binary omitted)"+text_bits[section]
    # we recognise ctrl-Z as end of file
    endoffile = text.find('\x1a')
    if endoffile >= 0: 
        text = text[:endoffile]
    split = string.split(text,'\n')
    if maxlength > 0:
        toolong = filter(lambda a:len(a)>maxlength,split)
        if toolong:
            pos = split.index(toolong[0])
            raise StarError( 'Line %d contains more than %d characters' % (pos+1,maxlength))
    # honour the header string
    if text[:10] != "#\#CIF_2.0" and ('2.0',Y20) in try_list:
        try_list.remove(('2.0',Y20),)
        if not try_list:
            raise StarError, 'File %s missing CIF2.0 header' % (filename)
    for grammar_name,Y in try_list:
       if scantype == 'standard' or grammar_name in ['2.0','STAR2']:
            parser = Y.StarParser(Y.StarParserScanner(text))
       else:
            parser = Y.StarParser(Y.yappsrt.Scanner(None,[],text,scantype='flex'))
       # handle encoding switch
       if grammar_name in ['2.0','STAR2']:
           prepared.set_characterset('unicode')
       else:
           prepared.set_characterset('ascii')
       proto_star = None
       try:
           proto_star = getattr(parser,"input")(prepared)
       except Y.yappsrt.SyntaxError,e:
           input = parser._scanner.input
           Y.yappsrt.print_error(input, e, parser._scanner)
       except Y.yappsrt.NoMoreTokens:
           print >>sys.stderr, 'Could not complete parsing; stopped around here:'
           print >>sys.stderr, parser._scanner
       except ValueError:
           pass
       if proto_star is not None: 
           proto_star.set_grammar(grammar_name)   #remember for output
           break
    if proto_star is None:
        errorstring = 'Syntax error in input file: last value parsed was %s' % Y.lastval
        errorstring = errorstring + '\nParser status: %s' % `parser._scanner`
        raise StarError( errorstring)
    # set visibility correctly
    proto_star.scoping = 'dictionary'
    proto_star.set_uri(my_uri)
    proto_star.scoping = save_scoping
    return proto_star

def get_dim(dataitem,current=0,packlen=0):
    zerotypes = [int, long, float, basestring]
    if type(dataitem) in zerotypes:
        return current, packlen
    if not dataitem.__class__ == ().__class__ and \
       not dataitem.__class__ == [].__class__:
       return current, packlen
    elif len(dataitem)>0: 
    #    print "Get_dim: %d: %s" % (current,`dataitem`)
        return get_dim(dataitem[0],current+1,len(dataitem))
    else: return current+1,0
    
def apply_line_folding(instring,minwraplength=60,maxwraplength=80):
    """Insert line folding characters into instring between min/max wraplength"""
    # first check that we need to do this
    lines = instring.split('\n')
    line_len = [len(l) for l in lines]
    if max(line_len) < maxwraplength and re.match("\\[ \v\t\f]*\n",instring) is None:
        return instring
    outstring = "\\\n"   #header
    for l in lines:
        if len(l) < maxwraplength:
            outstring = outstring + l
            if len(l) > 0 and l[-1]=='\\': #who'da thunk it?  A line ending with a backslash
                    outstring = outstring + "\\\n"  #
            outstring = outstring + "\n"  #  put back the split character
        else:
            current_bit = l
            while len(current_bit) > maxwraplength:
                space_pos = re.search('[ \v\f\t]+',current_bit[minwraplength:])
                if space_pos is not None and space_pos.start()<maxwraplength-1:
                    outstring = outstring + current_bit[:minwraplength+space_pos.start()] + "\\\n"
                    current_bit = current_bit[minwraplength+space_pos.start():]
                else:    #just blindly insert
                    outstring = outstring + current_bit[:maxwraplength-1] + "\\\n"
                    current_bit = current_bit[maxwraplength-1:]
            outstring = outstring + current_bit
            if current_bit[-1] == '\\':  #a backslash just happens to be here
                outstring = outstring + "\\\n"
            outstring = outstring + '\n'
    outstring = outstring[:-1]  #remove final newline
    return outstring

def remove_line_folding(instring):
    """Remove line folding from instring"""
    if re.match(r"\\[ \v\t\f]*" +"\n",instring) is not None:
        return re.sub(r"\\[ \v\t\f]*$" + "\n?","",instring,flags=re.M)
    else:
        return instring

def apply_line_prefix(instring,prefix):
    """Prefix every line in instring with prefix"""
    if prefix[0] != ";" and "\\" not in prefix:
        header = re.match(r"(\\[ \v\t\f]*" +"\n)",instring)
        if header is not None:
            print 'Found line folded string for prefixing...'
            not_header = instring[header.end():]
            outstring = prefix + "\\\\\n" + prefix
        else:
            print 'No folding in input string...'
            not_header = instring
            outstring = prefix + "\\\n" + prefix
        outstring = outstring + not_header.replace("\n","\n"+prefix)
        return outstring
    raise StarError, "Requested prefix starts with semicolon or contains a backslash: " + prefix

def remove_line_prefix(instring):
    """Remove prefix from every line if present"""
    prefix_match = re.match("(?P<prefix>[^;\\\n][^\n\\\\]+)(?P<folding>\\\\{1,2}[ \t\v\f]*\n)",instring)
    if prefix_match is not None:
        prefix_text = prefix_match.group('prefix')
        print 'Found prefix %s' % prefix_text
        prefix_end = prefix_match.end('folding')
        # keep any line folding instructions
        if prefix_match.group('folding')[:2]=='\\\\':  #two backslashes
            outstring = instring[prefix_match.end('folding')-1:].replace("\n"+prefix_text,"\n")
            return "\\" + outstring  #keep line folding first line
        else:
            outstring = instring[prefix_match.end('folding')-1:].replace("\n"+prefix_text,"\n")
            return outstring[1:]   #drop first line ending, no longer necessary
    else:
        return instring
    

def listify(item):
    if isinstance(item,basestring): return [item]
    else: return item

#Transpose the list of lists passed to us
def transpose(base_list):
    new_lofl = []
    full_length = len(base_list)
    opt_range = range(full_length)
    for i in range(len(base_list[0])):
       new_packet = [] 
       for j in opt_range:
          new_packet.append(base_list[j][i])
       new_lofl.append(new_packet)
    return new_lofl


def check_stringiness(data):
   """Check that the contents of data are all strings"""
   if not hasattr(data,'dtype'):   #so not Numpy
       from numbers import Number
       if isinstance(data,Number): return False
       elif isinstance(data,basestring): return True
       elif data is None:return False  #should be data are None :)
       else:
           for one_item in data:
               if not check_stringiness(one_item): return False
           return True   #all must be strings
   else:   #numerical python
       import numpy
       if data.ndim == 0:    #a bare value
           if data.dtype.kind in ['S','U']: return True
           else: return False
       else:
           for one_item in numpy.nditer(data):
               print 'numpy data: ' + `one_item`
               if not check_stringiness(one_item): return False
           return True

def process_template(template_file):
    """Process a template datafile to formatting instructions"""
    template_as_cif = StarFile(template_file,grammar="2.0").first_block()
    template_string = open(template_file).read()
    #template_as_lines = template_string.split("\n")
    #template_as_lines = [l for l in template_as_lines if len(l)>0 and l[0]!='#']
    #template_as_lines = [l for l in template_as_lines if l.split()[0] != 'loop_']
    #template_full_lines = dict([(l.split()[0],l) for l in template_as_lines if len(l.split())>0])
    form_hints = []   #ordered array of hint dictionaries
    for item in template_as_cif.item_order:  #order of input
        if not isinstance(item,int):    #not nested
            hint_dict = {"dataname":item}
            # find the line in the file
            start_pos = re.search("(^[ \t]*(?P<name>" + item + ")[ \t\n]+)(?P<spec>([\S]+)|(^;))",template_string,re.I|re.M)
            if start_pos.group("spec") != None:
                spec_pos = start_pos.start("spec")-start_pos.start(0)
                spec_char = template_string[start_pos.start("spec"):start_pos.start("spec")+3]
                if spec_char[0] in '\'";':
                    hint_dict.update({"delimiter":spec_char[0]})
                    if spec_char == '"""' or spec_char == "'''":
                        hint_dict.update({"delimiter":spec_char})
                if spec_char[0] != ";":   #so we need to work out the column number
                    hint_dict.update({"column":spec_pos})
                else:                  #need to put in the carriage return
                    hint_dict.update({"delimiter":"\n;"})
                if start_pos.group('name') != None:
                    name_pos = start_pos.start('name') - start_pos.start(0)
                    hint_dict.update({"name_pos":name_pos})
            #print '%s: %s' % (item,`hint_dict`)
            form_hints.append(hint_dict)
        else:           #loop block
            testnames = template_as_cif.loops[item]
            total_items = len(template_as_cif.loops[item])
            testname = testnames[0]
            #find the loop spec line in the file
            loop_regex = "(^[ \t]*(?P<loop>loop_)[ \t\n\r]+(?P<name>" + testname + ")([ \t\n\r]+_[\S]+){%d}[ \t]*$(?P<packet>(.(?!_loop|_[\S]+))*))" % (total_items - 1)
            loop_line = re.search(loop_regex,template_string,re.I|re.M|re.S)
            loop_so_far = loop_line.end()
            packet_text = loop_line.group('packet')
            loop_indent = loop_line.start('loop') - loop_line.start(0)
            form_hints.append({"dataname":'loop','name_pos':loop_indent})
            packet_regex = "[ \t]*(?P<all>(?P<sqqq>'''([^\n\r\f']*)''')|(?P<sq>'([^\n\r\f']*)'+)|(?P<dq>\"([^\n\r\"]*)\"+)|(?P<none>[^\s]+))"
            packet_pos = re.finditer(packet_regex,packet_text)
            line_end_pos = re.finditer("^",packet_text,re.M)
            next_end = line_end_pos.next().end()
            last_end = next_end
            for loopname in testnames:
                #find the name in the file for name pos
                name_regex = "(^[ \t]*(?P<name>" + loopname + "))"
                name_match = re.search(name_regex,template_string,re.I|re.M|re.S)
                loop_name_indent = name_match.start('name')-name_match.start(0)
                hint_dict = {"dataname":loopname,"name_pos":loop_name_indent}
                #find the value
                thismatch = packet_pos.next()
                while thismatch.start('all') > next_end:
                    try: 
                        last_end = next_end
                        next_end = line_end_pos.next().start()
                        print 'next end %d' % next_end
                    except StopIteration:
                        break
                print 'Start %d, last_end %d' % (thismatch.start('all'),last_end)
                col_pos = thismatch.start('all') - last_end + 1
                if thismatch.group('none') is None:
                    if thismatch.group('sqqq') is not None:
                        hint_dict.update({'delimiter':"'''"})
                    else:
                        hint_dict.update({'delimiter':thismatch.groups()[0][0]})
                hint_dict.update({'column':col_pos})
                print '%s: %s' % (loopname,`hint_dict`)
                form_hints.append(hint_dict)
    return form_hints


#No documentation flags

