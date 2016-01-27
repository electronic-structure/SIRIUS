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


import re
import StarFile
import sys

class CifBlock(StarFile.StarBlock):
    """
    A class to hold a single block of a CIF file.  A `CifBlock` object can be treated as
    a Python dictionary, in particular, individual items can be accessed using square
    brackets e.g. `b['_a_dataname']`.  All other Python dictionary methods are also
    available (e.g. `keys()`, `values()`).  Looped datanames will return a list of values.

    ## Initialisation

    When provided, `data` should be another `CifBlock` whose contents will be copied to
    this block.

    * if `strict` is set, maximum name lengths will be enforced

    * `maxoutlength` is the maximum length for output lines

    * `wraplength` is the ideal length to make output lines

    * When set, `overwrite` allows the values of datanames to be changed (otherwise an error
    is raised).

    * `compat_mode` will allow deprecated behaviour of creating single-dataname loops using
    the syntax `a[_dataname] = [1,2,3,4]`.  This should now be done by calling `CreateLoop`
    after setting the dataitem value.
    """
    def __init__(self,data = (), strict = 1, compat_mode=False, **kwargs):
        """When provided, `data` should be another CifBlock whose contents will be copied to
        this block.

        * if `strict` is set, maximum name lengths will be enforced

        * `maxoutlength` is the maximum length for output lines

        * `wraplength` is the ideal length to make output lines

        * When set, `overwrite` allows the values of datanames to be changed (otherwise an error
        is raised).

        * `compat_mode` will allow deprecated behaviour of creating single-dataname loops using
        the syntax `a[_dataname] = [1,2,3,4]`.  This should now be done by calling `CreateLoop`
        after setting the dataitem value.
        """
        if strict: maxnamelength=75
        else:
           maxnamelength=-1
        super(CifBlock,self).__init__(data=data,maxnamelength=maxnamelength,**kwargs)
        self.dictionary = None   #DDL dictionary referring to this block
        self.compat_mode = compat_mode   #old-style behaviour of setitem

    def RemoveCifItem(self,itemname): 
        """Remove `itemname` from the CifBlock"""
        self.RemoveItem(itemname)

    def __setitem__(self,key,value):
        self.AddItem(key,value)
        # for backwards compatibility make a single-element loop
        if self.compat_mode:
            if isinstance(value,(tuple,list)) and not isinstance(value,StarFile.StarList):
                 # single element loop
                 self.CreateLoop([key])

    def copy(self):
        newblock = super(CifBlock,self).copy()
        return self.copy.im_class(newblock)   #catch inheritance

    def AddCifItem(self,data):
        """ *DEPRECATED*. Use `AddItem` instead."""
        # we accept only tuples, strings and lists!!
        if not (isinstance(data[0],(basestring,tuple,list))):
                  raise TypeError, 'Cif datanames are either a string, tuple or list'
        # we catch single item loops as well...
        if isinstance(data[0],basestring):
            self.AddSingleCifItem(data[0],list(data[1]))
            if isinstance(data[1],(tuple,list)) and not isinstance(data[1],StarFile.StarList):  # a single element loop
                self.CreateLoop([data[0]])
            return
        # otherwise, we loop over the datanames
        keyvals = zip(data[0][0],[list(a) for a in data[1][0]])
        [self.AddSingleCifItem(a,b) for a,b in keyvals]
        # and create the loop
        self.CreateLoop(data[0][0])

    def AddSingleCifItem(self,key,value):
        """*Deprecated*. Use `AddItem` instead"""
        """Add a single data item. If it is part of a loop, a separate call should be made"""
        self.AddItem(key,value)

    def loopnames(self):
        return [self.loops[a] for a in self.loops]
                              

class CifFile(StarFile.StarFile):
    def __init__(self,datasource=None,strict=1,standard='CIF',**kwargs):
        super(CifFile,self).__init__(datasource=datasource,standard=standard, **kwargs)
        self.strict = strict
        self.header_comment = \
"""
##########################################################################
#               Crystallographic Information Format file 
#               Produced by PyCifRW module
# 
#  This is a CIF file.  CIF has been adopted by the International
#  Union of Crystallography as the standard for data archiving and 
#  transmission.
#
#  For information on this file format, follow the CIF links at
#  http://www.iucr.org
##########################################################################
"""


class CifError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return '\nCif Format error: '+ self.value 

class ValidCifError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return '\nCif Validity error: ' + self.value

class CifDic(StarFile.BlockCollection):
    """Create a Cif Dictionary object from the provided source, which can
    be a filename/URL or a CifFile.  Optional arguments:

    * do_minimum:
         Do not set up the dREL system for auto-calculation.  This is
         useful during recursive loads where the dREL links will be
         established at the end
    """
    def __init__(self,dic,do_minimum=False,grammar='1.1',**kwargs):
        self.do_minimum = do_minimum
        self.dic_as_cif = dic
        self.template_cache = {}    #for DDLm imports
        self.ddlm_functions = {}    #for DDLm functions
        self.switch_numpy(False)    #no Numpy arrays returned 
        if isinstance(dic,basestring):
            self.dic_as_cif = CifFile(dic,grammar=grammar,**kwargs)
            self.dic_as_cif.standard = 'Dic'    #for correct output order
            self.dic_as_cif.scoping = 'dictionary'
        (self.dicname,self.diclang,self.defdata) = self.dic_determine(self.dic_as_cif)
        print '%s is a %s dictionary' % (self.dicname,self.diclang)
        super(CifDic,self).__init__(datasource=self.defdata.copy(),**kwargs) 
        self.scopes_mandatory = {}
        self.scopes_naughty = {}
        self.grammar = grammar # remember for importing dictionaries correctly
        # rename and expand out definitions using "_name" in DDL dictionaries
        if self.diclang == "DDL1":
            self.DDL1_normalise()   #this removes any non-definition entries
            self.ddl1_cat_load()
        elif self.diclang == "DDL2":
            self.DDL2_normalise()   #iron out some DDL2 tricky bits
        elif self.diclang == "DDLm":
            self.scoping = 'dictionary'   #expose all save frames
            self.ddlm_normalise()
            self.ddlm_import()      #recursively calls this routine
            self.create_alias_table()
            self.create_cat_obj_table()
            self.create_cat_key_table()
            if not self.do_minimum:
                print "Doing full dictionary initialisation" 
                self.initialise_drel()
        self.add_category_info()
        # initialise type information
        self.typedic={}
        self.primdic = {}   #typecode<->primitive type translation
        self.add_type_info()
        if self.diclang != 'DDLm':
          self.item_validation_funs = [
            self.validate_item_type,
            self.validate_item_esd,
            self.validate_item_enum,   # functions which check conformance
            self.validate_enum_range,
            self.validate_looping]
          self.loop_validation_funs = [
            self.validate_loop_membership,
            self.validate_loop_key,
            self.validate_loop_references]    # functions checking loop values
          self.global_validation_funs = [
            self.validate_exclusion,
            self.validate_parent,
            self.validate_child,
            self.validate_dependents,
            self.validate_uniqueness] # where we need to look at other values
          self.block_validation_funs = [  # where only a full block will do
            self.validate_mandatory_category]
          self.global_remove_validation_funs = [
            self.validate_remove_parent_child] # removal is quicker with special checks
        elif self.diclang == 'DDLm':
            self.item_validation_funs = [
                self.validate_item_enum,
                ]
            self.loop_validation_funs = [
                self.validate_looping_ddlm,
                self.validate_loop_key_ddlm,
                self.validate_loop_membership
                ]
            self.global_validation_funs = []
            self.block_validation_funs = [
                self.check_mandatory_items,
                self.check_prohibited_items
                ]
            self.global_remove_validation_funs = []
        self.optimize = False        # default value
        self.done_parents = []
        self.done_children = []
        self.done_keys = []
        # debug
        # j = open("dic_debug","w")
        # j.write(self.__str__())
        # j.close()

    def dic_determine(self,cifdic):
        if cifdic.has_key("on_this_dictionary"): 
            self.master_key = "on_this_dictionary"
            self.type_spec = "_type"
            self.enum_spec = "_enumeration"
            self.cat_spec = "_category"
            self.esd_spec = "_type_conditions"
            self.must_loop_spec = "_list"
            self.must_exist_spec = "_list_mandatory"
            self.list_ref_spec = "_list_reference"
            self.key_spec = "_list_mandatory"
            self.unique_spec = "_list_uniqueness"
            self.child_spec = "_list_link_child"
            self.parent_spec = "_list_link_parent"
            self.related_func = "_related_function"
            self.related_item = "_related_item"
            self.primitive_type = "_type"
            self.dep_spec = "xxx"
            self.cat_list = []   #to save searching all the time
            name = cifdic["on_this_dictionary"]["_dictionary_name"]
            version = cifdic["on_this_dictionary"]["_dictionary_version"]
            return (name+version,"DDL1",cifdic)
        elif len(cifdic.get_roots()) == 1:              # DDL2/DDLm
            self.master_key = cifdic.get_roots()[0][0]      
            # now change to dictionary scoping
            cifdic.scoping = 'dictionary'
            name = cifdic[self.master_key]["_dictionary.title"]
            version = cifdic[self.master_key]["_dictionary.version"]
            if name != self.master_key:
                print "Warning: DDL2 blockname %s not equal to dictionary name %s" % (self.master_key,name)
            if cifdic[self.master_key].has_key("_dictionary.class"):   #DDLm
                self.enum_spec = '_enumeration_set.state'
                self.key_spec = '_category.key_id'
                self.must_exist_spec = None
                self.cat_spec = '_name.category_id'
                return(name+version,"DDLm",cifdic) 
            else:   #DDL2
                self.key_spec = "_category_mandatory.name"
                self.type_spec = "_item_type.code" 
                self.enum_spec = "_item_enumeration.value"
                self.esd_spec = "_item_type_conditions.code"
                self.cat_spec = "_item.category_id" 
                self.loop_spec = "there_is_no_loop_spec!"
                self.must_loop_spec = "xxx"
                self.must_exist_spec = "_item.mandatory_code"
                self.child_spec = "_item_linked.child_name"
                self.parent_spec = "_item_linked.parent_name"
                self.related_func = "_item_related.function_code"
                self.related_item = "_item_related.related_name"
                self.unique_spec = "_category_key.name"
                self.list_ref_spec = "xxx"
                self.primitive_type = "_type"
                self.dep_spec = "_item_dependent.dependent_name"
                return (name+version,"DDL2",cifdic)
        else:
            raise CifError, "Unable to determine dictionary DDL version"
        
    def DDL1_normalise(self):
        # switch off block name collision checks
        self.standard = None
        # add default type information in DDL2 style
        # initial types and constructs
        base_types = ["char","numb","null"]
        prim_types = base_types[:] 
        base_constructs = [".*",
            '(-?(([0-9]*[.][0-9]+)|([0-9]+)[.]?)([(][0-9]+[)])?([eEdD][+-]?[0-9]+)?)|\?|\.',
            "\"\" "]
        for key,value in self.items():
           newnames = [key]  #keep by default
           if value.has_key("_name"):
               real_name = value["_name"]
               if isinstance(real_name,list):        #looped values
                   for looped_name in real_name:
                      new_value = value.copy()
                      new_value["_name"] = looped_name  #only looped name
                      self[looped_name] = new_value
                   newnames = real_name
               else: 
                      self[real_name] = value
                      newnames = [real_name]
           # delete the old one
           if key not in newnames:
              del self[key]
        # loop again to normalise the contents of each definition
        for key,value in self.items():
           #unlock the block
           save_overwrite = value.overwrite
           value.overwrite = True
           # deal with a missing _list, _type_conditions
           if not value.has_key("_list"): value["_list"] = 'no'
           if not value.has_key("_type_conditions"): value["_type_conditions"] = 'none'
           # deal with enumeration ranges
           if value.has_key("_enumeration_range"):
               max,min = self.getmaxmin(value["_enumeration_range"])
               if min == ".":
                   self[key].AddLoopItem((("_item_range.maximum","_item_range.minimum"),((max,max),(max,min))))
               elif max == ".":
                   self[key].AddLoopItem((("_item_range.maximum","_item_range.minimum"),((max,min),(min,min))))
               else:
                   self[key].AddLoopItem((("_item_range.maximum","_item_range.minimum"),((max,max,min),(max,min,min))))
           #add any type construct information
           if value.has_key("_type_construct"):
               base_types.append(value["_name"]+"_type")   #ie dataname_type
               base_constructs.append(value["_type_construct"]+"$")
               prim_types.append(value["_type"])     #keep a record
               value["_type"] = base_types[-1]   #the new type name
               
        #make categories conform with ddl2
        #note that we must remove everything from the last underscore
           if value.get("_category",None) == "category_overview":
                last_under = value["_name"].rindex("_")
                catid = value["_name"][1:last_under]
                value["_category.id"] = catid  #remove square bracks
                if catid not in self.cat_list: self.cat_list.append(catid)
           value.overwrite = save_overwrite 
        # we now add any missing categories before filling in the rest of the
        # information
        for key,value in self.items():
            #print 'processing ddl1 definition %s' % key
            if self[key].has_key("_category"):
                if self[key]["_category"] not in self.cat_list:
                    # rogue category, add it in
                    newcat = self[key]["_category"]
                    fake_name = "_" + newcat + "_[]" 
                    newcatdata = CifBlock()
                    newcatdata["_category"] = "category_overview"
                    newcatdata["_category.id"] = newcat
                    newcatdata["_type"] = "null"
                    self[fake_name] = newcatdata
                    self.cat_list.append(newcat)
        # write out the type information in DDL2 style
        self.dic_as_cif[self.master_key].AddLoopItem((
            ("_item_type_list.code","_item_type_list.construct",
              "_item_type_list.primitive_code"),
            (base_types,base_constructs,prim_types)
            ))
     
    def create_pcloop(self,definition):
        old_children = self[definition].get('_item_linked.child_name',[])
        old_parents = self[definition].get('_item_linked.parent_name',[])
        if isinstance(old_children,basestring): 
             old_children = [old_children]
        if isinstance(old_parents,basestring): 
             old_parents = [old_parents]
        if (len(old_children)==0 and len(old_parents)==0) or \
           (len(old_children) > 1 and len(old_parents)>1):
             return
        if len(old_children)==0:
             old_children = [definition]*len(old_parents)
        if len(old_parents)==0:
             old_parents = [definition]*len(old_children)
        newloop = CifLoopBlock(dimension=1)
        newloop.AddLoopItem(('_item_linked.parent_name',old_parents)) 
        newloop.AddLoopItem(('_item_linked.child_name',old_children)) 
        try:
            del self[definition]['_item_linked.parent_name']
            del self[definition]['_item_linked.child_name']
        except KeyError:
            pass
        self[definition].insert_loop(newloop)
            
        

    def DDL2_normalise(self):
       listed_defs = filter(lambda a:isinstance(self[a].get('_item.name'),list),self.keys()) 
       # now filter out all the single element lists!
       dodgy_defs = filter(lambda a:len(self[a]['_item.name']) > 1, listed_defs)
       for item_def in dodgy_defs:
                # print "DDL2 norm: processing %s" % item_def
                thisdef = self[item_def]
                packet_no = thisdef['_item.name'].index(item_def)
                realcat = thisdef['_item.category_id'][packet_no] 
                realmand = thisdef['_item.mandatory_code'][packet_no]
                # first add in all the missing categories
                # we don't replace the entry in the list corresponding to the
                # current item, as that would wipe out the information we want
                for child_no in range(len(thisdef['_item.name'])):
                    if child_no == packet_no: continue
                    child_name = thisdef['_item.name'][child_no]
                    child_cat = thisdef['_item.category_id'][child_no]
                    child_mand = thisdef['_item.mandatory_code'][child_no]
                    if not self.has_key(child_name):
                        self[child_name] = CifBlock()
                        self[child_name]['_item.name'] = child_name
                    self[child_name]['_item.category_id'] = child_cat
                    self[child_name]['_item.mandatory_code'] = child_mand
                self[item_def]['_item.name'] = item_def
                self[item_def]['_item.category_id'] = realcat
                self[item_def]['_item.mandatory_code'] = realmand

       target_defs = filter(lambda a:self[a].has_key('_item_linked.child_name') or \
                                     self[a].has_key('_item_linked.parent_name'),self.keys())
       # now dodgy_defs contains all definition blocks with more than one child/parent link
       for item_def in dodgy_defs: self.create_pcloop(item_def)           #regularise appearance
       for item_def in dodgy_defs:
             print 'Processing %s' % item_def
             thisdef = self[item_def]
             child_list = thisdef['_item_linked.child_name']
             parents = thisdef['_item_linked.parent_name']
             # for each parent, find the list of children.
             family = zip(parents,child_list)
             notmychildren = family         #We aim to remove non-children
             # Loop over the parents, relocating as necessary
             while len(notmychildren):
                # get all children of first entry
                mychildren = filter(lambda a:a[0]==notmychildren[0][0],family)
                print "Parent %s: %d children" % (notmychildren[0][0],len(mychildren))
                for parent,child in mychildren:   #parent is the same for all
                         # Make sure that we simply add in the new entry for the child, not replace it,
                         # otherwise we might spoil the child entry loop structure
                         try:
                             childloop = self[child].GetLoop('_item_linked.parent_name')
                         except KeyError:
                             print 'Creating new parent entry %s for definition %s' % (parent,child)
                             self[child]['_item_linked.parent_name'] = [parent]
                             childloop = self[child].GetLoop('_item_linked.parent_name')
                             childloop.AddLoopItem(('_item_linked.child_name',[child]))
                             continue
                         else:
                             # A parent loop already exists and so will a child loop due to the
                             # call to create_pcloop above
                             pars = [a for a in childloop if getattr(a,'_item_linked.child_name','')==child]
                             goodpars = [a for a in pars if getattr(a,'_item_linked.parent_name','')==parent]
                             if len(goodpars)>0:   #no need to add it
                                 print 'Skipping duplicated parent - child entry in %s: %s - %s' % (child,parent,child)
                                 continue
                             print 'Adding %s to %s entry' % (parent,child)
                             newpacket = childloop.GetPacket(0)   #essentially a copy, I hope
                             setattr(newpacket,'_item_linked.child_name',child)
                             setattr(newpacket,'_item_linked.parent_name',parent)
                             childloop.AddPacket(newpacket)
                #
                # Make sure the parent also points to the children.  We get
                # the current entry, then add our 
                # new values if they are not there already
                # 
                parent_name = mychildren[0][0]
                old_children = self[parent_name].get('_item_linked.child_name',[])
                old_parents = self[parent_name].get('_item_linked.parent_name',[])
                oldfamily = zip(old_parents,old_children)
                newfamily = []
                print 'Old parents -> %s' % `old_parents`
                for jj, childname in mychildren:
                    alreadythere = filter(lambda a:a[0]==parent_name and a[1] ==childname,oldfamily)
                    if len(alreadythere)>0: continue
                    'Adding new child %s to parent definition at %s' % (childname,parent_name)
                    old_children.append(childname)
                    old_parents.append(parent_name)
                # Now output the loop, blowing away previous definitions.  If there is something
                # else in this category, we are destroying it.
                newloop = CifLoopBlock(dimension=1)
                newloop.AddLoopItem(('_item_linked.parent_name',old_parents))
                newloop.AddLoopItem(('_item_linked.child_name',old_children))
                del self[parent_name]['_item_linked.parent_name']
                del self[parent_name]['_item_linked.child_name']
                self[parent_name].insert_loop(newloop)
                print 'New parents -> %s' % `self[parent_name]['_item_linked.parent_name']`
                # now make a new,smaller list
                notmychildren = filter(lambda a:a[0]!=mychildren[0][0],notmychildren)

       # now flatten any single element lists
       single_defs = filter(lambda a:len(self[a]['_item.name'])==1,listed_defs)
       for flat_def in single_defs:
           flat_keys = self[flat_def].GetLoop('_item.name').keys()
           for flat_key in flat_keys: self[flat_def][flat_key] = self[flat_def][flat_key][0]
       # now deal with the multiple lists
       # next we do aliases
       all_aliases = filter(lambda a:self[a].has_key('_item_aliases.alias_name'),self.keys()) 
       for aliased in all_aliases:
          my_aliases = listify(self[aliased]['_item_aliases.alias_name'])
          for alias in my_aliases:
              self[alias] = self[aliased].copy()   #we are going to delete stuff...
              del self[alias]["_item_aliases.alias_name"]
 
    def ddlm_normalise(self):
        for key,value in self.items():
           if value.has_key("_definition.id"):
               real_name = value["_definition.id"]
               if real_name.lower() != key.lower():
                  self.rename(key,real_name)
        
    def ddlm_parse_valid(self):
        if not self.dic_as_cif[self.master_key].has_key("_dictionary_valid.application"):
            return
        for scope_pack in self.dic_as_cif[self.master_key].GetLoop("_dictionary_valid.application"):
            scope = getattr(scope_pack,"_dictionary_valid.application")
            valid_info = getattr(scope_pack,"_dictionary_valid.attributes")
            if scope[1] == "Mandatory":
                self.scopes_mandatory[scope[0]] = self.expand_category_opt(valid_info)
            elif scope[1] == "Prohibited":
                self.scopes_naughty[scope[0]] = self.expand_category_opt(valid_info)
                
    def expand_category_opt(self,name_list):
        """Return a list of all non-category items in a category or return the name
           if the name is not a category"""
        new_list = []
        for name in name_list:
          if self.get(name,{}).get('_definition.scope','Item') == 'Category':
            new_list += self.expand_category_opt([a for a in self.keys() if \
                     self[a].get('_name.category_id','').lower() == name.lower()])
          else:
            new_list.append(name)
        return new_list

    def ddlm_import(self):
        import urllib
        import_frames = [(a,self[a]['_import.get']) for a in self.keys() if self[a].has_key('_import.get')]
        #resolve all references
        for parent_block,import_list in import_frames:
          for import_ref in import_list:
            file_loc = import_ref["file"]
            full_uri = self.resolve_path(file_loc)
            if full_uri not in self.template_cache:
                dic_as_cif = CifFile(urllib.urlopen(full_uri),grammar=self.grammar)
                self.template_cache[full_uri] = CifDic(dic_as_cif,do_minimum=True)  #this will recurse internal imports
                print 'Added %s to cached dictionaries' % full_uri
            import_from = self.template_cache[full_uri]
            dupl = import_ref.get('dupl','Exit') 
            miss = import_ref.get('miss','Exit')
            target_key = import_ref["save"]
            try:
                import_target = import_from[target_key]
            except KeyError:
                if miss == 'Exit':
                   raise CifError,'Import frame %s not found in %s' % (target_key,full_uri)
                else: continue
            # now import appropriately
            mode = import_ref.get("mode",'Contents').lower()
            if self.has_key(target_key) and mode=='Full':  #so blockname will be duplicated
                if dupl == 'Exit':
                    raise CifError, 'Import frame %s already in dictionary' % target_key
                elif dupl == 'Ignore':
                    continue
            child_frames = import_from.get_children(target_key,include_parent=(mode=='full'))
            if mode == 'contents':   #merge self and children only
                # merge block contents
                self[parent_block].merge(import_target)
                self.merge_fast(child_frames,parent=parent_block)
            elif mode =="full":
                self.merge_fast(child_frames,parent=parent_block)      #
                print 'Merged %s (%d defs) in %s mode, now have %d defs' % (target_key,len(child_frames),
                   mode,len(self))
            # it will never happen again... 
            del self[self.master_key]["_import.get"]
                    
    def resolve_path(self,file_loc):
        import urlparse
        url_comps = urlparse.urlparse(file_loc)
        if url_comps[0]: return file_loc    #already full URI
        new_url = urlparse.urljoin(self.dic_as_cif.my_uri,file_loc)
        #print "Transformed %s to %s for import " % (file_loc,new_url)
        return new_url
        

    def create_alias_table(self):
        """Populate an alias table that we can look up when searching for a dataname"""
        all_aliases = [a for a in self.keys() if self[a].has_key('_alias.definition_id')]
        self.alias_table = dict([[a,self[a]['_alias.definition_id']] for a in all_aliases])

    def create_cat_obj_table(self):
        """Populate a table indexed by (cat,obj) and returning the correct dataname"""
        base_table = dict([((self[a].get('_name.category_id','').lower(),self[a].get('_name.object_id','').lower()),[self[a].get('_definition.id','')]) \
                           for a in self.keys() if self[a].get('_definition.scope','Item')=='Item'])
        self.loopable_cats = [a.lower() for a in self.keys() if self[a].get('_definition.class','')=='Loop']
        loopers = [self.get_immediate_children(a) for a in self.loopable_cats]
        loop_children = [[b[0] for b in a if b[0].lower() in self.loopable_cats ] for a in loopers]
        expand_list = dict([(a,b) for a,b in zip(self.loopable_cats,loop_children) if len(b)>0])
        print "Expansion list:" + `expand_list`
        extra_table = {}   #for debugging we keep it separate from base_table until the end
        def expand_base_table(parent_cat,child_cats):
            extra_names = []
            # first deal with all the child categories
            for child_cat in child_cats:
              nn = []
              if expand_list.has_key(child_cat):  # a nested category: grab its names
                nn = expand_base_table(child_cat,expand_list[child_cat])
                # store child names
                extra_names += nn
              # add all child names to the table
              child_names = [(self[n]['_name.object_id'].lower(),self[n]['_definition.id']) for n in self.names_in_cat(child_cat) if self[n].get('_definition.scope','Item')=='Item' and \
                                 self[n].get('_type.purpose','') != 'Key']
              child_names += extra_names
              extra_table.update(dict([((parent_cat,obj),[name]) for obj,name in child_names if (parent_cat,name) not in extra_table]))
            # and the repeated ones get appended instead
            repeats = [a for a in child_names if a in extra_table]
            for obj,name in repeats:
                extra_table[(parent_cat,obj)] += [name] 
            # and finally, add our own names to the return list
            child_names += [(self[n]['_name.object_id'].lower(),self[n]['_definition.id']) for n in self.names_in_cat(parent_cat) if self[n].get('_definition.scope','Item')=='Item' and \
                                self[n].get('_type.purpose','')!='Key']
            return child_names
        [expand_base_table(parent,child) for parent,child in expand_list.items()]
        print 'Expansion cat/obj values: ' + `extra_table`
        # append repeated ones
        non_repeats = dict([a for a in extra_table.items() if a[0] not in base_table])
        repeats = [a for a in extra_table.keys() if a in base_table]
        base_table.update(non_repeats)
        for k in repeats:
            base_table[k] += extra_table[k]
        self.cat_obj_lookup_table = base_table
        self.loop_expand_list = expand_list
        
    def create_cat_key_table(self):
        """Create a utility table with a list of keys applicable to each category"""
        self.cat_key_table = dict([(c,[self[c]["_category.key_id"]]) for c in self.loopable_cats])
        def collect_keys(parent_cat,child_cats):
                kk = []
                for child_cat in child_cats:
                    if self.loop_expand_list.has_key(child_cat):
                        kk += collect_keys(child_cat)
                    # add these keys to our list
                    kk += [self[child_cat]['_category.key_id']]
                self.cat_key_table[parent_cat] = self.cat_key_table[parent_cat] + kk
                return kk
        for k,v in self.loop_expand_list.items():
            collect_keys(k,v)
        print 'Keys for categories' + `self.cat_key_table`

    def ddl1_cat_load(self):
        deflist = self.keys()       #slight optimization
        cat_mand_dic = {}
        cat_unique_dic = {}
        # a function to extract any necessary information from each definition
        def get_cat_info(single_def):
            if self[single_def].get(self.must_exist_spec)=='yes':
                thiscat = self[single_def]["_category"]
                curval = cat_mand_dic.get(thiscat,[])
                curval.append(single_def)
                cat_mand_dic[thiscat] = curval
            # now the unique items...
            # cif_core.dic throws us a curly one: the value of list_uniqueness is
            # not the same as the defined item for publ_body_label, so we have
            # to collect both together.  We assume a non-listed entry, which
            # is true for all current (May 2005) ddl1 dictionaries.
            if self[single_def].get(self.unique_spec,None)!=None:
                thiscat = self[single_def]["_category"]
                new_unique = self[single_def][self.unique_spec]
                uis = cat_unique_dic.get(thiscat,[])
                if single_def not in uis: uis.append(single_def)
                if new_unique not in uis: uis.append(new_unique)
                cat_unique_dic[thiscat] = uis
            
        map(get_cat_info,deflist)       # apply the above function
        for cat in cat_mand_dic.keys():
            cat_entry = self.get_ddl1_entry(cat)
            self[cat_entry]["_category_mandatory.name"] = cat_mand_dic[cat]
        for cat in cat_unique_dic.keys():
            cat_entry = self.get_ddl1_entry(cat)
            self[cat_entry]["_category_key.name"] = cat_unique_dic[cat]

    # A helper function get find the entry corresponding to a given category name:
    # yes, in DDL1 the actual name is different in the category block due to the
    # addition of square brackets which may or may not contain stuff.

    def get_ddl1_entry(self,cat_name):
        chop_len = len(cat_name) 
        possibles = filter(lambda a:a[1:chop_len+3]==cat_name+"_[",self.keys())
        if len(possibles) > 1 or possibles == []:
            raise ValidCifError, "Category name %s can't be matched to category entry" % cat_name
        else:
            return possibles[0]

    def add_type_info(self):
        if self.dic_as_cif[self.master_key].has_key("_item_type_list.construct"): 
            types = self.dic_as_cif[self.master_key]["_item_type_list.code"]
            prim_types = self.dic_as_cif[self.master_key]["_item_type_list.primitive_code"]
            constructs = map(lambda a: a + "$", self.dic_as_cif[self.master_key]["_item_type_list.construct"])
            # add in \r wherever we see \n, and change \{ to \\{
            def regex_fiddle(mm_regex):
                brack_match = r"((.*\[.+)(\\{)(.*\].*))" 
                ret_match = r"((.*\[.+)(\\n)(.*\].*))" 
                fixed_regexp = mm_regex[:]  #copy
                # fix the brackets
                bm = re.match(brack_match,mm_regex)
                if bm != None: 
                    fixed_regexp = bm.expand(r"\2\\\\{\4")
                # fix missing \r
                rm = re.match(ret_match,fixed_regexp)
                if rm != None:
                    fixed_regexp = rm.expand(r"\2\3\\r\4")    
                #print "Regexp %s becomes %s" % (mm_regex,fixed_regexp)
                return fixed_regexp
            constructs = map(regex_fiddle,constructs)
            packed_up = map(None,types,constructs)
            for typecode,construct in packed_up:
                self.typedic[typecode] = re.compile(construct,re.MULTILINE|re.DOTALL)
            # now make a primitive <-> type construct mapping
            packed_up = map(None,types,prim_types)
            for typecode,primtype in packed_up:
                self.primdic[typecode] = primtype

    def add_category_info(self):
        if self.diclang == "DDLm":
            categories = [a for a in self.keys() if self[a].get("_definition.scope","Item")=="Category"]
            category_ids = [self[a]["_definition.id"] for a in categories]


            looped_cats = [a for a in categories if self[a].get('_definition.class','Set') == 'Loop']
            self.parent_lookup = {}
            for one_cat in looped_cats:
                parent_cat = one_cat
                parent_def = self[parent_cat]
                while self[parent_def['_name.category_id']].get('_definition.class','Set') == 'Loop':
                    parent_cat = parent_def['_name.category_id'].lower()
                    parent_def = self[parent_cat]
                self.parent_lookup[one_cat] = parent_cat

        else:
            categories = filter(lambda a:self[a].has_key("_category.id"),self.keys())
            # get the category id
            category_ids = map(lambda a:self[a]["_category.id"],categories)

            self.parent_lookup = dict(zip(categories,categories))  #dummy
        # match ids and entries in the dictionary
        catpairs = map(None,category_ids,categories)
        self.cat_map = dict(catpairs)

    def names_in_cat(self,cat,names_only=False):
        nameblocks = filter(lambda a:self[a].get("_name.category_id","").lower()
                             ==cat.lower(),self.keys())
        if not names_only:
            return ["_" + self[a]["_name.category_id"]+"." + self[a]["_name.object_id"] for a in nameblocks if self[a].get('_definition.scope','Item')=='Item']
        else:
            return map(lambda a:self[a]["_name.object_id"],nameblocks)
        
    def change_category_name(self,oldname,newname):
        self.unlock()
        """Change the category name from [[oldname]] to [[newname]]"""
        if not self.has_key(oldname):
            raise KeyError,'Cannot rename non-existent category %s to %s' % (oldname,newname)
        if self.has_key(newname):
            raise KeyError,'Cannot rename %s to %s as %s already exists' % (oldname,newname,oldname)
        self.rename(oldname,newname)   #NB no name integrity checks
        self[newname]['_name.object_id']=newname
        self[newname]['_definition.id']=newname
        child_defs = [a[0] for a in self.get_immediate_children(newname)]
        for child_def in child_defs:
            self[child_def]['_name.category_id'] = newname
            if self[child_def].get('_definition.scope','Item')=='Item':
                newid = self.create_catobj_name(newname,self[child_def]['_name.object_id'])
                self[child_def]['_definition.id']=newid
                self.rename(child_def,newid[1:])  #no underscore at the beginning
        # update categories
        print `self.cat_map.values()`
        oldid = [a[0] for a in self.cat_map.items() if a[1].lower()==oldname.lower()]
        if len(oldid)!=1:
            raise CifError, 'Unable to find old category name in category map: %s not in %s' % (oldname.lower(),`self.cat_map.items()`)
        del self.cat_map[oldid[0]]
        self.cat_map[newname.lower()] = newname
        self.lock()

    def create_catobj_name(self,cat,obj):
        """Combine category and object in approved fashion to create id"""
        return ('_'+cat+'.'+obj)

    def change_category(self,itemname,catname):
        """Move itemname into catname"""
        defid = self[itemname]
        if defid['_name.category_id'].lower()==catname.lower():
            print 'Already in category, no change'
            return
        if catname not in self:    #don't have it
            print 'No such category %s' % catname
            return
        self.unlock()
        objid = defid['_name.object_id']
        defid['_name.category_id'] = catname
        newid = itemname # stays the same for categories
        if defid.get('_definition.scope','Item') == 'Item':
            newid = self.create_catobj_name(catname,objid)
            defid['_definition.id']= newid
            self.rename(itemname,newid)
        self.set_parent(catname,newid)  
        self.lock()

    def change_name(self,one_def,newobj):
        """Change the object_id of one_def to newobj"""
        newid = self.create_catobj_name(self[one_def]['_name.category_id'],newobj)
        self.unlock()
        self.rename(one_def,newid)
        self[newid]['_definition.id']=newid
        self[newid]['_name.object_id']=newobj
        self.lock()
        return newid
       
    def add_category(self,catname,catparent=None):
        """Add a new category to the dictionary with name [[catname]].
           If [[catparent]] is None, the category will be a child of
           the topmost 'Head' category or else the top data block."""
        if catname in self:
            raise CifError, 'Attempt to add existing category %s' % catname
        self.unlock()
        if catparent is None:
            root_cat = [a for a in self.keys() if self[a].get('_definition.class',None)=='Head']
            if len(root_cat)>0:
                root_cat = root_cat[0]
            else:
                root_cat = self.get_roots()[0]
        else:
            root_cat = catparent
        realname = self.NewBlock(catname,parent=root_cat)
        self[realname]['_name.object_id'] = realname
        self[realname]['_name.category_id'] = self[root_cat]['_name.object_id']
        self[realname]['_definition.id'] = realname
        self[realname]['_definition.scope'] = 'Category'
        self[realname]['_definition.class'] = 'Loop'
        self[realname]['_description.text'] = 'No definition provided'
        self.lock()
        self.cat_map[realname]=realname
        return realname

    def add_definition(self,itemname,catparent,def_text='PLEASE DEFINE ME'):
        """Add itemname to category [[catparent]]. If itemname contains periods,
        all text before the final period is ignored."""
        self.unlock()
        if '.' in itemname:
            objname = itemname[:-1].split('.')[-1]
        else:
            objname = itemname
        objname = objname.strip('_')
        if catparent not in self or self[catparent]['_definition.scope']!='Category':
            raise CifError, 'No category %s in dictionary' % catparent
        fullname = '_'+catparent.lower()+'.'+objname
        print 'New name: %s' % fullname
        realname = self.NewBlock(fullname, fix=False, parent=catparent)
        self[realname]['_definition.id']=fullname
        self[realname]['_name.object_id']=objname
        self[realname]['_name.category_id']=catparent
        self[realname]['_definition.class']='Datum'
        self[realname]['_description.text']=def_text
        return realname
        
    def remove_definition(self,defname):
        """Remove a definition from the dictionary. If a category, we have to
        remove the links in cat_map"""
        if defname not in self:
            return
        if self[defname].get('_definition.scope')=='Category':
            child_cats = [a['_name.category_id'] for a in self.get_children(defname) if a.get('_definition.scope')=='Category']
            for cc in child_cats: del self.cat_map[cc]
        del self[defname]

    def get_cat_obj(self,name):
        """Return (cat,obj) tuple. [[name]] must contain only a single period"""
        cat,obj = name.split('.')
        return (cat.strip('_'),obj)
        
    def get_name_by_cat_obj(self,category,object,give_default=False):
        """Return the dataname corresponding to the given category and object"""
        if category[0] == '_':    #accidentally left in
           true_cat = category[1:].lower()
        else:
           true_cat = category.lower()
        try:
            return self.cat_obj_lookup_table[(true_cat,object.lower())][0]
        except KeyError:
            if give_default:
               return '_'+true_cat+'.'+object
        raise KeyError, 'No such category,object in the dictionary: %s %s' % (true_cat,object)


    def WriteOut(self,**kwargs):
        def drop_underscore(a):
            if a[0] == '_':
                return a[1:]
            else:
                return a
        myblockorder = self.get_full_child_list()
        # drop the underscores as the CIF will not have them
        myblockorder = [drop_underscore(a) for a in myblockorder]
        self.dic_as_cif.set_grammar(self.grammar)
        self.dic_as_cif.standard = 'Dic'
        return self.dic_as_cif.WriteOut(blockorder = myblockorder,**kwargs)

    def SetTemplate(self,template_file):
        self.dic_as_cif.SetTemplate(template_file)

    def get_full_child_list(self):
        """Return a list of definition blocks in order parent-child-child-child-parent-child..."""
        top_block = self.get_roots()[0][0]
        root_cat = [a for a in self.keys() if self[a].get('_definition.class',None)=='Head']
        if len(root_cat) == 1:
            all_names = [top_block] + self.recurse_child_list(root_cat[0])
            print 'Final block order: ' + `all_names`
            return all_names
        raise ValueError, 'Dictionary contains no Head category, please print as plain CIF instead'

    def recurse_child_list(self,parentname):
        """Recursively expand the logical child list of [[parentname]]"""
        final_list = [parentname]
        child_blocks = [a for a in self.child_table.keys() if self[a].get('_name.category_id','').lower() == parentname.lower()]
        child_blocks.sort()    #we love alphabetical order
        child_items = [a for a in child_blocks if self[a].get('_definition.scope','Item') == 'Item']
        final_list += child_items
        child_cats = [a for a in child_blocks if self[a].get('_definition.scope','Item') == 'Category']
        for child_cat in child_cats:
            final_list += self.recurse_child_list(child_cat)
        return final_list

        

    def get_key_pack(self,category,value,data):
        keyname = self[category][self.unique_spec]
        onepack = data.GetPackKey(keyname,value)
        return onepack
     
    def get_number_with_esd(numstring):
        import string
        numb_re = '((-?(([0-9]*[.]([0-9]+))|([0-9]+)[.]?))([(][0-9]+[)])?([eEdD][+-]?[0-9]+)?)|(\?)|(\.)' 
        our_match = re.match(numb_re,numstring)
        if our_match:
            a,base_num,b,c,dad,dbd,esd,exp,q,dot = our_match.groups()
        #    print "Debug: %s -> %s" % (numstring, `our_match.groups()`)
        else:
            return None,None
        if dot or q: return None,None     #a dot or question mark
        if exp:          #has exponent 
           exp = string.replace(exp,"d","e")     # mop up old fashioned numbers
           exp = string.replace(exp,"D","e")
           base_num = base_num + exp
        #print "Debug: have %s for base_num from %s" % (base_num,numstring)
        base_num = float(base_num)
        # work out esd, if present.
        if esd:
            esd = float(esd[1:-1])    # no brackets
            if dad:                   # decimal point + digits
                esd = esd * (10 ** (-1* len(dad)))
            if exp:
                esd = esd * (10 ** (float(exp[1:])))
        return base_num,esd

    def getmaxmin(self,rangeexp):
        regexp = '(-?(([0-9]*[.]([0-9]+))|([0-9]+)[.]?)([eEdD][+-]?[0-9]+)?)*' 
        regexp = regexp + ":" + regexp
        regexp = re.match(regexp,rangeexp)
        try:
            minimum = regexp.group(1)
            maximum = regexp.group(7)
        except AttributeError:
            print "Can't match %s" % rangeexp
        if minimum == None: minimum = "." 
        else: minimum = float(minimum)
        if maximum == None: maximum = "." 
        else: maximum = float(maximum)
        return maximum,minimum

    def initialise_drel(self):
        """Parse drel functions and prepare data structures in dictionary"""
        self.ddlm_parse_valid() #extract validity information from data block
        self.transform_drel()   #parse the drel functions
        self.add_drel_funcs()   #put the drel functions into the namespace

    def transform_drel(self):
        from drel import drel_ast_yacc
        from drel import py_from_ast
        import traceback
        parser = drel_ast_yacc.parser
        my_namespace = self.keys()
        my_namespace = dict(map(None,my_namespace,my_namespace)) 
        # we provide a table of loopable categories {cat_name:(key,[item_name,...]),...})
        loopable_cats = [a for a in self.keys() if self[a].get("_definition.class","Set")=="Loop"]
        loop_keys = [self[a]["_category.key_id"].split(".")[1] for a in loopable_cats]
        cat_names = [self.names_in_cat(a,names_only=True) for a in loopable_cats]
        loop_info = dict(zip(loopable_cats,zip(loop_keys,cat_names)))
        # parser.listable_items = [a for a in self.keys() if "*" in self[a].get("_type.dimension","")] 
        derivable_list = [a for a in self.keys() if self[a].has_key("_method.expression") \
                              and self[a].get("_definition.scope","")!='Category' \
                              and self[a].get("_name.category_id","")!= "function"]
        for derivable in derivable_list:
            target_id = derivable
            # reset the list of visible names for parser
            special_ids = [dict(map(None,self.keys(),self.keys()))]
            print "Target id: %s" % derivable
            drel_expr = self[derivable]["_method.expression"]
            if isinstance(drel_expr,list):
               try:
                    drel_expr = self[derivable].GetKeyedPacket("_method.purpose","Evaluation")
                    drel_expr = getattr(drel_expr,'_method.expression')
               except (KeyError,ValueError):
                    print 'No evaluation method found, skipping' 
                    continue
            # print "Transforming %s" % drel_expr
            # List categories are treated differently...
            try:
                meth_ast = parser.parse(drel_expr+"\n")
            except:
                print 'Syntax error in method for %s; leaving as is' % derivable
                a,b = sys.exc_info()[:2]
                print `a`,`b`
                print traceback.print_tb(sys.exc_info()[-1],None,sys.stdout)
                continue
            # Construct the python method
            pyth_meth = py_from_ast.make_python_function(meth_ast,"pyfunc",target_id,
                                                                       loopable=loop_info,
                                                         cif_dic = self)
            save_overwrite = self[derivable].overwrite
            self[derivable].overwrite = True
            self[derivable]["_method.py_expression"] = pyth_meth
            self[derivable].overwrite = save_overwrite
            #print "Final result:\n " + self[derivable]["_method.py_expression"]

    def add_drel_funcs(self):
        from drel import drel_ast_yacc
        from drel import py_from_ast
        funclist = [a for a in self.keys() if self[a].get("_name.category_id","")=='function']
        funcnames = [(self[a]["_name.object_id"],
                      getattr(self[a].GetKeyedPacket("_method.purpose","Evaluation"),"_method.expression")) for a in funclist]
        # create executable python code...
        parser = drel_ast_yacc.parser
        # we provide a table of loopable categories {cat_name:(key,[item_name,...]),...})
        loopable_cats = [a for a in self.keys() if self[a].get("_definition.class","Set")=="Loop"]
        loop_keys = [self[a]["_category.key_id"].split(".")[1] for a in loopable_cats]
        cat_names = [self.names_in_cat(a,names_only=True) for a in loopable_cats]
        loop_info = dict(zip(loopable_cats,zip(loop_keys,cat_names)))
        for funcname,funcbody in funcnames:
            parser.target_id = funcname
            res_ast = parser.parse(funcbody)
            py_function = py_from_ast.make_python_function(res_ast,None,targetname=funcname,func_def=True,loopable=loop_info,cif_dic = self)
            #print 'dREL library function ->\n' + py_function
            global_table = globals()
            # global_table.update(self.ddlm_functions)
            exec py_function in global_table    #add to namespace
        #print "All functions -> " + `self.ddlm_functions`
 
    def switch_numpy(self,to_val):
        pass

    def derive_item(self,start_key,cifdata,store_value = False):
        key = start_key   #starting value
        result = None     #success is a non-None value
        # check for aliases
        # check for an older form of a new value
        found_it = [k for k in self.alias_table.get(key,[]) if cifdata.has_key(k)]
        if len(found_it)>0:
            corrected_type = self.change_type(key,cifdata[found_it[0]])
            return corrected_type
        # now do the reverse check - any alternative form
        alias_name = [a for a in self.alias_table.items() if key in a[1]]
        print 'Aliases for %s: %s' % (key,`alias_name`)
        if len(alias_name)==1:
            key = alias_name[0][0]   #actual definition name
            if cifdata.has_key(key): return self.change_type(key,cifdata[key])
            found_it = [k for k in alias_name[0][1] if cifdata.has_key(k)]
            if len(found_it)>0:
                return self.change_type(key,cifdata[found_it[0]])
        elif len(alias_name)>1:
            raise CifError, 'Dictionary error: dataname alias appears in different definitions: ' + `alias_name`

        # store any default value in case we have a problem
        def_val = self[key].get("_enumeration.default","")
        def_index_val = self[key].get("_enumeration.def_index_id","")
        the_func = self[key].get('_method.py_expression',"")
        if the_func:   #attempt to calculate it
            #global_table = globals()
            #global_table.update(self.ddlm_functions)
            print 'Executing function for %s:' % key
            #print the_func
            exec the_func in globals(),locals() #will access dREL functions, puts "pyfunc" in scope
            # print 'in following global environment: ' + `global_table`
            stored_setting = cifdata.provide_value
            cifdata.provide_value = True
            result = pyfunc(cifdata)
            cifdata.provide_value = stored_setting
            #print "Function returned %s" % `result`

        if result is None:   # try defaults
            if def_val: return self.change_type(key,def_val)
            if def_index_val:            #derive a default value
                index_vals = self[key]["_enumeration_default.index"]
                val_to_index = cifdata[def_index_val]     #what we are keying on
                if self[def_index_val]['_type.contents'] in ['Code','Name','Tag']:
                    lcase_comp = True
                    index_vals = [a.lower() for a in index_vals]
                # Handle loops
                if isinstance(val_to_index,list):
                    if lcase_comp:
                        val_to_index = [a.lower() for a in val_to_index]
                    keypos = [index_vals.index(a) for a in val_to_index]
                    result = map(lambda a:self[key]["_enumeration_default.value"][a] ,keypos)
                else:
                    if lcase_comp:
                        val_to_index = val_to_index.lower()
                    keypos = index_vals.index(val_to_index)   #value error if no such value available
                    result = self[key]["_enumeration_default.value"][keypos]
                result = self.change_type(key,result)
                print "Indexed on %s to get %s for %s" % (def_index_val,`result`,`val_to_index`)

        # read it in
        if result is None:   #can't do anything else
            print 'Warning: no way of deriving item %s' % key
        # now try to insert the new information into the right place
        # find if items of this category already appear...
        if store_value: 
              # try to change any matrices etc. to lists
              the_category = self[key]["_name.category_id"]
              if self[the_category].get('_definition.class','Set')=='Loop':
                  # our result is looped, this only gets the simple structures
                  if result is not None and hasattr(result[0],'dtype'):    #numpy object
                      if result[0].size > 1:   #so is not a float
                          out_result = [StarFile.StarList(a.tolist()) for a in result]
                      else:
                          def conv_from_numpy(maybe_numpy):
                              try:
                                  return maybe_numpy.item(0)
                              except:
                                  return maybe_numpy
                          out_result = [conv_from_numpy(a) for a in result]
                  else:
                      out_result = result
                  # so out_result now contains a value suitable for storage
                  cat_names = [a for a in self.keys() if self[a].get("_name.category_id",None)==the_category]
                  has_cat_names = [a for a in cat_names if cifdata.has_key(a)]
                  print 'Found pre-existing names: ' + `has_cat_names`
                  if len(has_cat_names)>0:   #this category already exists
                      if out_result is None: #need to create a list of Nones
                          loop_len = len(cifdata[has_cat_names[0]])
                          out_result = [None]*loop_len
                          result = out_result
                  cifdata[key] = out_result      #lengths must match or else!!
                  print 'Loop info:' + `cifdata.loops`
                  cifdata.AddLoopName(has_cat_names[0],key)
              else:    #not a looped category 
                  if result is not None and hasattr(result,'dtype'):
                      if result.size > 1:
                          out_result = StarFile.StarList(result.tolist())
                          cifdata[key] = out_result
                      else:
                          cifdata[key] = result.item(0)
                  else:
                      cifdata[key] = result
        return result

    def change_type(self,itemname,inval):
        import numpy
        if inval == "?": return inval
        change_function = convert_type(self[itemname])
        if isinstance(inval,list) and not isinstance(inval,StarFile.StarList):   #from a loop
            newval = map(change_function,inval)
        else: 
            newval = change_function(inval)
        return newval

    def validate_item_type(self,item_name,item_value):
        def mymatch(m,a):  
            res = m.match(a)
            if res != None: return res.group() 
            else: return ""
        target_type = self[item_name].get(self.type_spec) 
        if target_type == None:          # e.g. a category definition
            return {"result":True}                  # not restricted in any way
        matchexpr = self.typedic[target_type]
        item_values = listify(item_value)
        #for item in item_values:
            #print "Type match " + item_name + " " + item + ":",
        #skip dots and question marks
        check_all = filter(lambda a: a !="." and a != "?",item_values)
        check_all = filter(lambda a: mymatch(matchexpr,a) != a, check_all)
        if len(check_all)>0: return {"result":False,"bad_values":check_all}
        else: return {"result":True}

    def decide(self,result_list):
        """Construct the return list"""
        if len(result_list)==0:
               return {"result":True}
        else:
               return {"result":False,"bad_values":result_list}

    def validate_item_container(self, item_name,item_value):
        container_type = self[item_name]['_type.container']
        item_values = listify(item_value)
        if container_type == 'Single':
           okcheck = [a for a in item_values if not isinstance(a,(int,float,long,basestring))]
           return decide(okcheck)
        if container_type in ('Multiple','List'):
           okcheck = [a for a in item_values if not isinstance(a,StarList)]
           return decide(okcheck)
        if container_type == 'Array':    #A list with numerical values
           okcheck = [a for a in item_values if not isinstance(a,StarList)]
           first_check = decide(okcheck)
           if not first_check['result']: return first_check
           #num_check = [a for a in item_values if len([b for b in a if not isinstance

    def validate_item_esd(self,item_name,item_value):
        if self[item_name].get(self.primitive_type) != 'numb':
            return {"result":None}
        can_esd = self[item_name].get(self.esd_spec,"none") == "esd" 
        if can_esd: return {"result":True}         #must be OK!
        item_values = listify(item_value)
        check_all = filter(lambda a: get_number_with_esd(a)[1] != None, item_values)
        if len(check_all)>0: return {"result":False,"bad_values":check_all}
        return {"result":True}

    def validate_enum_range(self,item_name,item_value):
        if not self[item_name].has_key("_item_range.minimum") and \
           not self[item_name].has_key("_item_range.maximum"):
            return {"result":None}
        minvals = self[item_name].get("_item_range.minimum",default = ["."])
        maxvals = self[item_name].get("_item_range.maximum",default = ["."])
        def makefloat(a):
            if a == ".": return a
            else: return float(a)
        maxvals = map(makefloat, maxvals)
        minvals = map(makefloat, minvals)
        rangelist = map(None,minvals,maxvals)
        item_values = listify(item_value)
        def map_check(rangelist,item_value):
            if item_value == "?" or item_value == ".": return True
            iv,esd = get_number_with_esd(item_value)
            if iv==None: return None  #shouldn't happen as is numb type
            for lower,upper in rangelist:
                #check the minima
                if lower == ".": lower = iv - 1
                if upper == ".": upper = iv + 1
                if iv > lower and iv < upper: return True
                if upper == lower and iv == upper: return True
            # debug
            # print "Value %s fails range check %d < x < %d" % (item_value,lower,upper)
            return False
        check_all = filter(lambda a,b=rangelist: map_check(b,a) != True, item_values)
        if len(check_all)>0: return {"result":False,"bad_values":check_all}
        else: return {"result":True}
                
    def validate_item_enum(self,item_name,item_value):
        try: 
            enum_list = self[item_name][self.enum_spec][:]
        except KeyError:
            return {"result":None}
        enum_list.append(".")   #default value
        enum_list.append("?")   #unknown
        item_values = listify(item_value)
        #print "Enum check: %s in %s" % (`item_values`,`enum_list`)
        check_all = [a for a in item_values if a not in enum_list]
        if len(check_all)>0: return {"result":False,"bad_values":check_all}
        else: return {"result":True}

    def validate_looping(self,item_name,item_value):
        try:
            must_loop = self[item_name][self.must_loop_spec]
        except KeyError:
            return {"result":None}
        if must_loop == 'yes' and isinstance(item_value,basestring): # not looped
            return {"result":False}      #this could be triggered
        if must_loop == 'no' and not isinstance(item_value,basestring): 
            return {"result":False}
        return {"result":True}

    def validate_looping_ddlm(self,loop_names):
        """Check that all names are loopable"""
        truly_loopy = self.get_final_cats(loop_names)
        if len(truly_loopy)<len(loop_names):  #some are bad
            categories = [(a,self[a][self.cat_spec].lower()) for a in loop_names]
            not_looped = [a[0] for a in categories if a[1] not in self.parent_lookup.keys()]
            return {"result":False,"bad_items":not_looped}
        return {"result":True}


    def validate_loop_membership(self,loop_names):
        final_cat = self.get_final_cats(loop_names)
        bad_items =  [a for a in final_cat if a != final_cat[0]]
        if len(bad_items)>0:
            return {"result":False,"bad_items":bad_items}
        else: return {"result":True}

    def get_final_cats(self,loop_names):
        """Return a list of the uppermost parent categories for the loop_names. Names
        that are not from loopable categories are ignored."""
        try:
            categories = [self[a][self.cat_spec].lower() for a in loop_names]
        except KeyError:       #category is mandatory
            raise ValidCifError( "%s missing from dictionary %s for item in loop containing %s" % (self.cat_spec,self.dicname,loop_names[0]))
        truly_looped = [a for a in categories if a in self.parent_lookup.keys()]
        return [self.parent_lookup[a] for a in truly_looped]

    def validate_loop_key(self,loop_names):
        category = self[loop_names[0]][self.cat_spec]
        # find any unique values which must be present 
        entry_name = self.cat_map[category]
        key_spec = self[entry_name].get(self.key_spec,[])
        for names_to_check in key_spec:
            if isinstance(names_to_check,basestring):   #only one
                names_to_check = [names_to_check]
            for loop_key in names_to_check:
                if loop_key not in loop_names: 
                    #is this one of those dang implicit items?
                    if self[loop_key].get(self.must_exist_spec,None) == "implicit":
                        continue          #it is virtually there...
                    alternates = self.get_alternates(loop_key)
                    if alternates == []: 
                        return {"result":False,"bad_items":loop_key}
                    for alt_names in alternates:
                        alt = filter(lambda a:a in loop_names,alt_names)
                        if len(alt) == 0: 
                            return {"result":False,"bad_items":loop_key}  # no alternates   
        return {"result":True}

    def validate_loop_key_ddlm(self,loop_names):
        """Make sure at least one of the necessary keys are available"""
        final_cats = self.get_final_cats(loop_names)
        if len(final_cats)>0:
            poss_keys = self.cat_key_table[final_cats[0]]
            found_keys = [a for a in poss_keys if a in loop_names]
            if len(found_keys)>0:
                return {"result":True}
            else:
                return {"result":False,"bad_items":poss_keys}
        else: 
            return {"result":True}

    def validate_loop_references(self,loop_names):
        must_haves = map(lambda a:self[a].get(self.list_ref_spec,None),loop_names)
        must_haves = filter(lambda a:a != None,must_haves)
        # build a flat list.  For efficiency we don't remove duplicates,as
        # we expect no more than the order of 10 or 20 looped names.
        def flat_func(a,b): 
            if isinstance(b,basestring): 
               a.append(b)       #single name
            else:
               a.extend(b)       #list of names
            return a
        flat_mh = reduce(flat_func,must_haves,[])
        group_mh = filter(lambda a:a[-1]=="_",flat_mh)
        single_mh = filter(lambda a:a[-1]!="_",flat_mh)
        res = filter(lambda a: a not in loop_names,single_mh)
        def check_gr(s_item, name_list):
            nl = map(lambda a:a[:len(s_item)],name_list)
            if s_item in nl: return True
            return False
        res_g = filter(lambda a:check_gr(a,loop_names),group_mh)
        if len(res) == 0 and len(res_g) == 0: return {"result":True}
        # construct alternate list
        alternates = map(lambda a: (a,self.get_alternates(a)),res)
        alternates = filter(lambda a:a[1] != [], alternates)
        # next two lines purely for error reporting
        missing_alts = filter(lambda a: a[1] == [], alternates)
        missing_alts = map(lambda a:a[0],missing_alts)
        if len(alternates) != len(res): 
           return {"result":False,"bad_items":missing_alts}   #short cut; at least one
                                                       #doesn't have an altern
        #loop over alternates
        for orig_name,alt_names in alternates:
             alt = filter(lambda a:a in loop_names,alt_names)
             if len(alt) == 0: return {"result":False,"bad_items":orig_name}# no alternates   
        return {"result":True}        #found alternates
             
    def get_alternates(self,main_name,exclusive_only=False):
        alternates = self[main_name].get(self.related_func,None)
        alt_names = []
        if alternates != None: 
            alt_names =  self[main_name].get(self.related_item,None)
            if isinstance(alt_names,basestring): 
                alt_names = [alt_names]
                alternates = [alternates]
            together = map(None,alt_names,alternates)
            if exclusive_only:
                alt_names = filter(lambda a:a[1]=="alternate_exclusive" \
                                             or a[1]=="replace", together)
            else:
                alt_names = filter(lambda a:a[1]=="alternate" or a[1]=="replace",together)
            alt_names = map(lambda a:a[0],alt_names)
        # now do the alias thing
        alias_names = listify(self[main_name].get("_item_aliases.alias_name",[]))
        alt_names.extend(alias_names)
        # print "Alternates for %s: %s" % (main_name,`alt_names`)
        return alt_names
        

    def validate_exclusion(self,item_name,item_value,whole_block,provisional_items={},globals={}):
       alternates = map(lambda a:a.lower(),self.get_alternates(item_name,exclusive_only=True))
       item_name_list = map(lambda a:a.lower(),whole_block.keys())
       item_name_list.extend(map(lambda a:a.lower(),provisional_items.keys()))
       bad = filter(lambda a:a in item_name_list,alternates)
       if len(bad)>0:
           print "Bad: %s, alternates %s" % (`bad`,`alternates`)
           return {"result":False,"bad_items":bad}
       else: return {"result":True}

    # validate that parent exists and contains matching values
    def validate_parent(self,item_name,item_value,whole_block,provisional_items={},globals={}):
        parent_item = self[item_name].get(self.parent_spec)
        if not parent_item: return {"result":None}   #no parent specified
        if isinstance(parent_item,list): 
            parent_item = parent_item[0]
        if self.optimize:
            if parent_item in self.done_parents:
                return {"result":None}
            else: 
                self.done_parents.append(parent_item)
                print "Done parents %s" % `self.done_parents`
        # initialise parent/child values
        if isinstance(item_value,basestring):
            child_values = [item_value]
        else: child_values = item_value[:]    #copy for safety
        # track down the parent
        # print "Looking for %s parent item %s in %s" % (item_name,parent_item,`whole_block`)
        # if globals contains the parent values, we are doing a DDL2 dictionary, and so 
        # we have collected all parent values into the global block - so no need to search
        # for them elsewhere. 
        # print "Looking for %s" % `parent_item`
        parent_values = globals.get(parent_item)
        if not parent_values:
            parent_values = provisional_items.get(parent_item,whole_block.get(parent_item))
        if not parent_values:  
            # go for alternates
            namespace = whole_block.keys()
            namespace.extend(provisional_items.keys())
            namespace.extend(globals.keys())
            alt_names = filter_present(self.get_alternates(parent_item),namespace)
            if len(alt_names) == 0:
                if len(filter(lambda a:a != "." and a != "?",child_values))>0:
                    return {"result":False,"parent":parent_item}#no parent available -> error
                else:
                    return {"result":None}       #maybe True is more appropriate??
            parent_item = alt_names[0]           #should never be more than one?? 
            parent_values = provisional_items.get(parent_item,whole_block.get(parent_item))
            if not parent_values:   # check global block
                parent_values = globals.get(parent_item)
        if isinstance(parent_values,basestring):
            parent_values = [parent_values]   
        #print "Checking parent %s against %s, values %s/%s" % (parent_item,
        #                                          item_name,`parent_values`,`child_values`)
        missing = self.check_parent_child(parent_values,child_values)
        if len(missing) > 0:
            return {"result":False,"bad_values":missing,"parent":parent_item}
        return {"result":True}

    def validate_child(self,item_name,item_value,whole_block,provisional_items={},globals={}):
        try:
            child_items = self[item_name][self.child_spec][:]  #copy
        except KeyError:
            return {"result":None}    #not relevant
        # special case for dictionaries  -> we check parents of children only
        if globals.has_key(item_name):  #dictionary so skip
            return {"result":None}
        if isinstance(child_items,basestring): # only one child
            child_items = [child_items]
        if isinstance(item_value,basestring): # single value
            parent_values = [item_value]
        else: parent_values = item_value[:]
        # expand child list with list of alternates
        for child_item in child_items[:]:
            child_items.extend(self.get_alternates(child_item))
        # now loop over the children
        for child_item in child_items:
            if self.optimize:
                if child_item in self.done_children:
                    return {"result":None}
                else: 
                    self.done_children.append(child_item)
                    print "Done children %s" % `self.done_children`
            if provisional_items.has_key(child_item):
                child_values = provisional_items[child_item][:]
            elif whole_block.has_key(child_item):
                child_values = whole_block[child_item][:]
            else:  continue 
            if isinstance(child_values,basestring):
                child_values = [child_values]
            #    print "Checking child %s against %s, values %s/%s" % (child_item,
            #                                          item_name,`child_values`,`parent_values`)
            missing = self.check_parent_child(parent_values,child_values)
            if len(missing)>0:
                return {"result":False,"bad_values":missing,"child":child_item}
        return {"result":True}       #could mean that no child items present
           
    #a generic checker: all child vals should appear in parent_vals
    def check_parent_child(self,parent_vals,child_vals):
        # shield ourselves from dots and question marks
        pv = parent_vals[:]
        pv.extend([".","?"])
        res =  filter(lambda a:a not in pv,child_vals)
        #print "Missing: %s" % res
        return res

    def validate_remove_parent_child(self,item_name,whole_block):
        try:
            child_items = self[item_name][self.child_spec]
        except KeyError:
            return {"result":None}
        if isinstance(child_items,basestring): # only one child
            child_items = [child_items]
        for child_item in child_items:
            if whole_block.has_key(child_item): 
                return {"result":False,"child":child_item}
        return {"result":True}
         
    def validate_dependents(self,item_name,item_value,whole_block,prov={},globals={}):
        try:
            dep_items = self[item_name][self.dep_spec][:]
        except KeyError:
            return {"result":None}    #not relevant
        if isinstance(dep_items,basestring):
            dep_items = [dep_items]
        actual_names = whole_block.keys()
        actual_names.extend(prov.keys())
        actual_names.extend(globals.keys())
        missing = filter(lambda a:a not in actual_names,dep_items)
        if len(missing) > 0:
            alternates = map(lambda a:[self.get_alternates(a),a],missing)
            # compact way to get a list of alternative items which are 
            # present
            have_check = map(lambda b:[filter_present(b[0],actual_names),
                                       b[1]],alternates) 
            have_check = filter(lambda a:len(a[0])==0,have_check)
            if len(have_check) > 0:
                have_check = map(lambda a:a[1],have_check)
                return {"result":False,"bad_items":have_check}
        return {"result":True}
        
    def validate_uniqueness(self,item_name,item_value,whole_block,provisional_items={},
                                                                  globals={}):
        category = self[item_name].get(self.cat_spec)
        if category == None:
            print "No category found for %s" % item_name
            return {"result":None}
        # print "Category %s for item %s" % (`category`,item_name)
        catentry = self.cat_map[category]
        # we make a copy in the following as we will be removing stuff later!
        unique_i = self[catentry].get("_category_key.name",[])[:]
        if isinstance(unique_i,basestring):
            unique_i = [unique_i]
        if item_name not in unique_i:       #no need to verify
            return {"result":None}
        if isinstance(item_value,basestring):  #not looped
            return {"result":None}
        # print "Checking %s -> %s -> %s ->Unique: " % (item_name,category,catentry) + `unique_i`
        # check that we can't optimize by not doing this check
        if self.optimize:
            if unique_i in self.done_keys:
                return {"result":None}
            else:
                self.done_keys.append(unique_i)
        val_list = []
        # get the matching data from any other data items
        unique_i.remove(item_name)
        other_data = []
        if len(unique_i) > 0:            # i.e. do have others to think about
           for other_name in unique_i:
           # we look for the value first in the provisional dict, then the main block
           # the logic being that anything in the provisional dict overrides the
           # main block
               if provisional_items.has_key(other_name):
                   other_data.append(provisional_items[other_name]) 
               elif whole_block.has_key(other_name):
                   other_data.append(whole_block[other_name])
               elif self[other_name].get(self.must_exist_spec)=="implicit":
                   other_data.append([item_name]*len(item_value))  #placeholder
               else:
                   return {"result":False,"bad_items":other_name}#missing data name
        # ok, so we go through all of our values
        # this works by comparing lists of strings to one other, and
        # so could be fooled if you think that '1.' and '1' are 
        # identical
        for i in range(len(item_value)):
            #print "Value no. %d" % i ,
            this_entry = item_value[i]
            for j in range(len(other_data)):
                this_entry = " ".join([this_entry,other_data[j][i]]) 
            #print "Looking for %s in %s: " % (`this_entry`,`val_list`)
            if this_entry in val_list: 
                return {"result":False,"bad_values":this_entry}
            val_list.append(this_entry)
        return {"result":True}


    def validate_mandatory_category(self,whole_block):
        mand_cats = filter(lambda a:self[a].get("_category.mandatory_code","no")=="yes",
                    self.keys())
        # map to actual ids
        catlist = self.cat_map.items()
        # print "Mandatory categories - %s" % `mand_cats`
        all_keys = whole_block.keys() #non-save block keys
        for mand_cat in mand_cats:
            cat_id = filter(lambda a:a[1]==mand_cat,catlist)[0][0]
            no_of_items = len(filter(lambda a:self[a].get(self.cat_spec)==cat_id,
                                 all_keys))
            if no_of_items == 0:
                return {"result":False,"bad_items":cat_id}
        return {"result":True}

    def find_prob_cats(self,whole_block):
        mand_cats = filter(lambda a:self[a].get("_category.mandatory_code","no")=="yes",
                    self.keys())
        # map to actual ids
        catlist = self.cat_map.items()
        # find missing categories
        wbs = whole_block["saves"]
        abs_all_keys = whole_block.keys()
        abs_all_keys.extend(reduce(lambda a,b:a+(wbs[b].keys()),wbs.keys(),[]))
        prob_cats = []
        for mand_cat in mand_cats:
            cat_id = filter(lambda a:a[1]==mand_cat,catlist)[0][0]
            
            if len(filter(lambda a:self[a].get(self.cat_spec)==cat_id,abs_all_keys))==0:
                prob_cats.append(cat_id)
        if len(prob_cats) > 0:
            return (False,{'whole_block':[('validate_mandatory_category',{"result":False,"bad_items":problem_cats})]})
        else:
            return (True,{})

    def check_mandatory_items(self,whole_block,default_scope='Item'):
        """Return an error if any mandatory items are missing"""
        if len(self.scopes_mandatory)== 0: return {"result":True}
        if default_scope == 'Datablock':
            return {"result":True}     #is a data file
        scope = whole_block.get('_definition.scope',default_scope)
        missing = [a for a in self.scopes_mandatory[scope] if not whole_block.has_key(a)]
        if len(missing)==0:
            return {"result":True}
        else:
            return {"result":False,"bad_items":missing}

    def check_prohibited_items(self,whole_block,default_scope='Item'):
        """Return an error if any prohibited items are present"""
        if len(self.scopes_naughty)== 0: return {"result":True}
        if default_scope == 'Datablock':
            return {"result":True}     #is a data file
        scope = whole_block.get('_definition.scope',default_scope)
        present = [a for a in self.scopes_naughty[scope] if whole_block.has_key(a)]
        if len(present)==0:
            return {"result":True}
        else:
            return {"result":False,"bad_items":present}



    def run_item_validation(self,item_name,item_value):
        return {item_name:map(lambda f:(f.__name__,f(item_name,item_value)),self.item_validation_funs)}

    def run_loop_validation(self,loop_names):
        return {loop_names[0]:map(lambda f:(f.__name__,f(loop_names)),self.loop_validation_funs)}

    def run_global_validation(self,item_name,item_value,data_block,provisional_items={},globals={}):
        results = map(lambda f:(f.__name__,f(item_name,item_value,data_block,provisional_items,globals)),self.global_validation_funs)
        return {item_name:results}

    def run_block_validation(self,whole_block,block_scope='Item'):
        results = [(f.__name__,f(whole_block)) for f in self.block_validation_funs]
        # fix up the return values
        return {"whole_block":results}

    def optimize_on(self):
        self.optimize = True
        self.done_keys = []
        self.done_children = []
        self.done_parents = []

    def optimize_off(self):
        self.optimize = False
        self.done_keys = []
        self.done_children = []
        self.done_parents = []


class ValidCifBlock(CifBlock):
    """A `CifBlock` that is valid with respect to a given CIF dictionary.  Methods
    of `CifBlock` are overridden where necessary to disallow addition of invalid items to the
    `CifBlock`.

    ## Initialisation
 
    * `dic` is a `CifDic` object to be used for validation.

    """
    def __init__(self,dic = None, diclist=[], mergemode = "replace",*args,**kwords):
        CifBlock.__init__(self,*args,**kwords)    
        if dic and diclist:
            print "Warning: diclist argument ignored when initialising ValidCifBlock"
        if isinstance(dic,CifDic):
            self.fulldic = dic
        else:
            raise TypeError( "ValidCifBlock passed non-CifDic type in dic argument")
        if len(diclist)==0 and not dic:
            raise ValidCifError( "At least one dictionary must be specified")
        if diclist and not dic:
            self.fulldic = merge_dic(diclist,mergemode)
        if not self.run_data_checks()[0]:
            raise ValidCifError( self.report())

    def run_data_checks(self,verbose=False):
        self.v_result = {}
        self.fulldic.optimize_on()
        for dataname in self.keys():
            update_value(self.v_result,self.fulldic.run_item_validation(dataname,self[dataname]))
            update_value(self.v_result,self.fulldic.run_global_validation(dataname,self[dataname],self))
        for loop_names in self.loops.values():
            update_value(self.v_result,self.fulldic.run_loop_validation(loop_names))
        # now run block-level checks
        update_value(self.v_result,self.fulldic.run_block_validation(self))
        # return false and list of baddies if anything didn't match
        self.fulldic.optimize_off()
        for test_key in self.v_result.keys():
            #print "%s: %s" % (test_key,`self.v_result[test_key]`)
            self.v_result[test_key] = filter(lambda a:a[1]["result"]==False,self.v_result[test_key])
            if len(self.v_result[test_key]) == 0: 
                del self.v_result[test_key]
        isvalid = len(self.v_result)==0
        #if not isvalid:
        #    print "Baddies:" + `self.v_result`
        return isvalid,self.v_result

    def single_item_check(self,item_name,item_value):
        #self.match_single_item(item_name)
        if not self.fulldic.has_key(item_name):
            result = {item_name:[]}
        else:
            result = self.fulldic.run_item_validation(item_name,item_value)
        baddies = filter(lambda a:a[1]["result"]==False, result[item_name])
        # if even one false one is found, this should trigger
        isvalid = (len(baddies) == 0)
        # if not isvalid: print "Failures for %s:" % item_name + `baddies`
        return isvalid,baddies

    def loop_item_check(self,loop_names):
        in_dic_names = filter(lambda a:self.fulldic.has_key(a),loop_names)
        if len(in_dic_names)==0:
            result = {loop_names[0]:[]}
        else:
            result = self.fulldic.run_loop_validation(in_dic_names)
        baddies = filter(lambda a:a[1]["result"]==False,result[in_dic_names[0]])
        # if even one false one is found, this should trigger
        isvalid = (len(baddies) == 0)
        # if not isvalid: print "Failures for %s:" % `loop_names` + `baddies`
        return isvalid,baddies

    def global_item_check(self,item_name,item_value,provisional_items={}):
        if not self.fulldic.has_key(item_name):
            result = {item_name:[]}
        else:
            result = self.fulldic.run_global_validation(item_name,
               item_value,self,provisional_items = provisional_items)
        baddies = filter(lambda a:a[1]["result"]==False,result[item_name])
        # if even one false one is found, this should trigger
        isvalid = (len(baddies) == 0)
        # if not isvalid: print "Failures for %s:" % item_name + `baddies`
        return isvalid,baddies

    def remove_global_item_check(self,item_name):
        if not self.fulldic.has_key(item_name):
            result = {item_name:[]}
        else:
            result = self.fulldic.run_remove_global_validation(item_name,self,False)
        baddies = filter(lambda a:a[1]["result"]==False,result[item_name])
        # if even one false one is found, this should trigger
        isvalid = (len(baddies) == 0)
        # if not isvalid: print "Failures for %s:" % item_name + `baddies`
        return isvalid,baddies

    def AddToLoop(self,dataname,loopdata):
        # single item checks
        paired_data = loopdata.items()
        for name,value in paired_data:
            valid,problems = self.single_item_check(name,value) 
            self.report_if_invalid(valid,problems)
        # loop item checks; merge with current loop
        found = 0
        for aloop in self.block["loops"]:
            if aloop.has_key(dataname):
                loopnames = aloop.keys()
                for new_name in loopdata.keys():
                    if new_name not in loopnames: loopnames.append(new_name)
                valid,problems = self.looped_item_check(loopnames)
                self.report_if_invalid(valid,problems)
        prov_dict = loopdata.copy()
        for name,value in paired_data: 
            del prov_dict[name]   # remove temporarily
            valid,problems = self.global_item_check(name,value,prov_dict)
            prov_dict[name] = value  # add back in
            self.report_if_invalid(valid,problems)
        CifBlock.AddToLoop(self,dataname,loopdata)
 
    def AddCifItem(self,data):
        if isinstance(data[0],basestring):   # single item
            valid,problems = self.single_item_check(data[0],data[1])
            self.report_if_invalid(valid,problems,data[0])
            valid,problems = self.global_item_check(data[0],data[1])
            self.report_if_invalid(valid,problems,data[0])
        elif isinstance(data[0],tuple) or isinstance(data[0],list):
            paired_data = map(None,data[0],data[1])
            for name,value in paired_data:
                valid,problems = self.single_item_check(name,value) 
                self.report_if_invalid(valid,problems,name)
            valid,problems = self.loop_item_check(data[0])
            self.report_if_invalid(valid,problems,data[0])
            prov_dict = {}            # for storing temporary items
            for name,value in paired_data: prov_dict[name]=value
            for name,value in paired_data: 
                del prov_dict[name]   # remove temporarily
                valid,problems = self.global_item_check(name,value,prov_dict)
                prov_dict[name] = value  # add back in
                self.report_if_invalid(valid,problems,name)
        super(ValidCifBlock,self).AddCifItem(data)

    def AddItem(self,key,value,**kwargs):
        """Set value of dataname `key` to `value` after checking for conformance with CIF dictionary"""
        valid,problems = self.single_item_check(key,value)
        self.report_if_invalid(valid,problems,key)
        valid,problems = self.global_item_check(key,value)
        self.report_if_invalid(valid,problems,key)
        super(ValidCifBlock,self).AddItem(key,value,**kwargs)

    # utility function
    def report_if_invalid(self,valid,bad_list,data_name):
        if not valid:
            error_string = reduce(lambda a,b: a + "," + b[0], bad_list, "") 
            error_string = `data_name` + " fails following validity checks: "  + error_string
            raise ValidCifError( error_string)

    def __delitem__(self,key):
        # we don't need to run single item checks; we do need to run loop and
        # global checks.
        if self.has_key(key):
            try: 
                loop_items = self.GetLoop(key)
            except TypeError:
                loop_items = []
            if loop_items:             #need to check loop conformance
                loop_names = map(lambda a:a[0],loop_items)
                loop_names = filter(lambda a:a != key,loop_names)
                valid,problems = self.loop_item_check(loop_names)
                self.report_if_invalid(valid,problems)
            valid,problems = self.remove_global_item_check(key)
            self.report_if_invalid(valid,problems)
        self.RemoveCifItem(key)


    def report(self):
       import cStringIO
       outstr = cStringIO.StringIO()
       outstr.write( "Validation results\n")
       outstr.write( "------------------\n")
       print "%d invalid items found\n" % len(self.v_result)
       for item_name,val_func_list in self.v_result.items():
           outstr.write("%s fails following tests:\n" % item_name)
           for val_func in val_func_list:
               outstr.write("\t%s\n")
       return outstr.getvalue()


class ValidCifFile(CifFile):
    """A CIF file for which all datablocks are valid.  Argument `dic` to
    initialisation specifies a `CifDic` object to use for validation."""
    def __init__(self,dic=None,diclist=[],mergemode="replace",*args,**kwargs):
        if not diclist and not dic and not hasattr(self,'bigdic'):
            raise ValidCifError( "At least one dictionary is required to create a ValidCifFile object")
        if not dic and diclist:     #merge here for speed
            self.bigdic = merge_dic(diclist,mergemode)
        elif dic and not diclist:
            self.bigdic = dic
        CifFile.__init__(self,*args,**kwargs)
        for blockname in self.keys():
            self.dictionary[blockname]=ValidCifBlock(data=self.dictionary[blockname],dic=self.bigdic)

    def NewBlock(self,blockname,blockcontents,**kwargs):
        CifFile.NewBlock(self,blockname,blockcontents,**kwargs)
        # dictionary[blockname] is now a CifBlock object.  We
        # turn it into a ValidCifBlock object
        self.dictionary[blockname] = ValidCifBlock(dic=self.bigdic,
                                         data=self.dictionary[blockname])


class ValidationResult:
    """Represents validation result. It is initialised with """
    def __init__(self,results):
        """results is return value of validate function"""
        self.valid_result, self.no_matches = results

    def report(self,use_html):
        """Return string with human-readable description of validation result"""
        return validate_report((self.valid_result, self.no_matches),use_html)

    def is_valid(self,block_name=None):
        """Return True for valid CIF file, otherwise False"""
        if block_name is not None:
            block_names = [block_name]
        else:
            block_names = self.valid_result.iterkeys()
        for block_name in block_names:
            if not self.valid_result[block_name] == (True,{}):
                valid = False
                break
            else:
                valid = True
        return valid
    
    def has_no_match_items(self,block_name=None):
        """Return true if some items are not found in dictionary"""
        if block_name is not None:
            block_names = [block_name]
        else:
            block_names = self.no_matches.iter_keys() 
        for block_name in block_names:
            if self.no_matches[block_name]:
                has_no_match_items = True
                break
            else:
                has_no_match_items = False
        return has_no_match_items
    

        
def Validate(ciffile,dic = "", diclist=[],mergemode="replace",isdic=False):
    """Validate the `ciffile` conforms to the definitions in `CifDic` object `dic`, or if `dic` is missing,
    to the results of merging the `CifDic` objects in `diclist` according to `mergemode`.  Flag
    `isdic` indicates that `ciffile` is a CIF dictionary meaning that save frames should be
    accessed for validation and that mandatory_category should be interpreted differently for DDL2."""
    check_file = CifFile(ciffile)
    if not dic:
        fulldic = merge_dic(diclist,mergemode)
    else:
        fulldic = dic
    no_matches = {}
    valid_result = {}
    if isdic:          #assume one block only
        check_file.scoping = 'instance' #only data blocks visible
        top_level = check_file.keys()[0]
        check_file.scoping = 'dictionary'   #all blocks visible
        # collect a list of parents for speed
        if fulldic.diclang == 'DDL2':
            poss_parents = fulldic.get_all("_item_linked.parent_name")
            for parent in poss_parents:
                curr_parent = listify(check_file.get(parent,[]))
                new_vals = check_file.get_all(parent)
                new_vals.extend(curr_parent)
                if len(new_vals)>0:
                    check_file[parent] = new_vals
                print "Added %s (len %d)" % (parent,len(check_file[parent]))
            # next dictionary problem: the main DDL2 dictionary has what
            # I would characterise as a mandatory_category problem, but
            # in order to gloss over it, we allow a different 
            # interpretation, which requires only a single check for one
            # block.
            #if fake_mand:
            #    valid_result[blockname] = fulldic.find_prob_cats(check_file)
            #    no_matches[blockname] = [a for a in check_file.keys if not fulldic.has_key(a)]
    # now run the validations
    for block in check_file.keys(): 
        if isdic and block == top_level:
           block_scope = 'Dictionary'
        elif isdic:
           block_scope = 'Item'
        else:
           block_scope = 'Datablock'
        no_matches[block] = [a for a in check_file[block].keys() if not fulldic.has_key(a)]
        # remove non-matching items
        print "Not matched: " + `no_matches[block]`
        for nogood in no_matches[block]:
             del check_file[block][nogood]
        print "Validating block %s, scope %s" % (block,block_scope)
        valid_result[block] = run_data_checks(check_file[block],fulldic,block_scope=block_scope)
    return valid_result,no_matches

def validate_report(val_result,use_html=False):
    import cStringIO
    valid_result,no_matches = val_result
    outstr = cStringIO.StringIO()
    if use_html:
        outstr.write("<h2>Validation results</h2>")
    else:
        outstr.write( "Validation results\n")
        outstr.write( "------------------\n")
    if len(valid_result) > 10:  
        suppress_valid = True         #don't clutter with valid messages
        if use_html:
           outstr.write("<p>For brevity, valid blocks are not reported in the output.</p>")
    else:
        suppress_valid = False
    for block in valid_result.keys():
        block_result = valid_result[block]
        if block_result[0]:
            out_line = "Block '%s' is VALID" % block
        else:
            out_line = "Block '%s' is INVALID" % block
        if use_html:
            if (block_result[0] and (not suppress_valid or len(no_matches[block])>0)) or not block_result[0]:
                outstr.write( "<h3>%s</h3><p>" % out_line)
        else:
                outstr.write( "\n %s\n" % out_line)
        if len(no_matches[block])!= 0:
            if use_html:
                outstr.write( "<p>The following items were not found in the dictionary")
                outstr.write(" (note that this does not invalidate the data block):</p>")
                outstr.write("<p><table>\n")
                map(lambda it:outstr.write("<tr><td>%s</td></tr>" % it),no_matches[block])
                outstr.write("</table>\n")
            else:
                outstr.write( "\n The following items were not found in the dictionary:\n")
                outstr.write("Note that this does not invalidate the data block\n")
                map(lambda it:outstr.write("%s\n" % it),no_matches[block])
        # now organise our results by type of error, not data item...
        error_type_dic = {}
        for error_item, error_list in block_result[1].items():
            for func_name,bad_result in error_list:
                bad_result.update({"item_name":error_item})
                try:
                    error_type_dic[func_name].append(bad_result)
                except KeyError:
                    error_type_dic[func_name] = [bad_result]
        # make a table of test name, test message
        info_table = {\
        'validate_item_type':\
            "The following data items had badly formed values",
        'validate_item_esd':\
            "The following data items should not have esds appended",
        'validate_enum_range':\
            "The following data items have values outside permitted range",
        'validate_item_enum':\
            "The following data items have values outside permitted set",
        'validate_looping':\
            "The following data items violate looping constraints",
        'validate_loop_membership':\
            "The following looped data names are of different categories to the first looped data name",
        'validate_loop_key':\
            "A required dataname for this category is missing from the loop\n containing the dataname",
        'validate_loop_key_ddlm':\
            "A loop key is missing for the category containing the dataname",
        'validate_loop_references':\
            "A dataname required by the item is missing from the loop",
        'validate_parent':\
            "A parent dataname is missing or contains different values",
        'validate_child':\
            "A child dataname contains different values to the parent",
        'validate_uniqueness':\
            "One or more data items do not take unique values",
        'validate_dependents':\
            "A dataname required by the item is missing from the data block",
        'validate_exclusion': \
            "Both dataname and exclusive alternates or aliases are present in data block",
        'validate_mandatory_category':\
            "A required category is missing from this block",
        'check_mandatory_items':\
            "A required data attribute is missing from this block",
        'check_prohibited_items':\
            "A prohibited data attribute is present in this block"}

        for test_name,test_results in error_type_dic.items():
           if use_html:
               outstr.write(html_error_report(test_name,info_table[test_name],test_results)) 
           else:
               outstr.write(error_report(test_name,info_table[test_name],test_results)) 
               outstr.write("\n\n")
    return outstr.getvalue()
         
# A function to lay out a single error report.  We are passed
# the name of the error (one of our validation functions), the
# explanation to print out, and a dictionary with the error 
# information.  We print no more than 50 characters of the item

def error_report(error_name,error_explanation,error_dics):
   retstring = "\n\n " + error_explanation + ":\n\n"
   headstring = "%-32s" % "Item name"
   bodystring = ""
   if error_dics[0].has_key("bad_values"):
      headstring += "%-20s" % "Bad value(s)"
   if error_dics[0].has_key("bad_items"):
      headstring += "%-20s" % "Bad dataname(s)"
   if error_dics[0].has_key("child"):
      headstring += "%-20s" % "Child"
   if error_dics[0].has_key("parent"):
      headstring += "%-20s" % "Parent" 
   headstring +="\n"
   for error in error_dics:
      bodystring += "\n%-32s" % error["item_name"]
      if error.has_key("bad_values"):
          out_vals = map(lambda a:`a`[:50],error["bad_values"])
          bodystring += "%-20s" % out_vals 
      if error.has_key("bad_items"):
          bodystring += "%-20s" % `error["bad_items"]`
      if error.has_key("child"):
          bodystring += "%-20s" % `error["child"]`
      if error.has_key("parent"):
          bodystring += "%-20s" % `error["parent"]`
   return retstring + headstring + bodystring 

#  This lays out an HTML error report

def html_error_report(error_name,error_explanation,error_dics,annotate=[]):
   retstring = "<h4>" + error_explanation + ":</h4>"
   retstring = retstring + "<table cellpadding=5><tr>"
   headstring = "<th>Item name</th>"
   bodystring = ""
   if error_dics[0].has_key("bad_values"):
      headstring += "<th>Bad value(s)</th>"
   if error_dics[0].has_key("bad_items"):
      headstring += "<th>Bad dataname(s)</th>"
   if error_dics[0].has_key("child"):
      headstring += "<th>Child</th>"
   if error_dics[0].has_key("parent"):
      headstring += "<th>Parent</th>" 
   headstring +="</tr>\n"
   for error in error_dics:
      bodystring += "<tr><td><tt>%s</tt></td>" % error["item_name"]
      if error.has_key("bad_values"):
          bodystring += "<td>%s</td>" % error["bad_values"]
      if error.has_key("bad_items"):
          bodystring += "<td><tt>%s</tt></td>" % error["bad_items"]
      if error.has_key("child"):
          bodystring += "<td><tt>%s</tt></td>" % error["child"]
      if error.has_key("parent"):
          bodystring += "<td><tt>%s</tt></td>" % error["parent"]
      bodystring += "</tr>\n"
   return retstring + headstring + bodystring + "</table>\n"

def run_data_checks(check_block,fulldic,block_scope='Item'):
    v_result = {}
    for key in check_block.keys():
        update_value(v_result, fulldic.run_item_validation(key,check_block[key]))
        update_value(v_result, fulldic.run_global_validation(key,check_block[key],check_block))
    for loopnames in check_block.loops.values():
        update_value(v_result, fulldic.run_loop_validation(loopnames))
    update_value(v_result,fulldic.run_block_validation(check_block,block_scope=block_scope))
    # return false and list of baddies if anything didn't match
    for test_key in v_result.keys():
        v_result[test_key] = filter(lambda a:a[1]["result"]==False,v_result[test_key])
        if len(v_result[test_key]) == 0: 
            del v_result[test_key]
    # if even one false one is found, this should trigger
    # print "Baddies:" + `v_result`
    isvalid = len(v_result)==0
    return isvalid,v_result
    

def get_number_with_esd(numstring):
    import string
    numb_re = '((-?(([0-9]*[.]([0-9]+))|([0-9]+)[.]?))([(][0-9]+[)])?([eEdD][+-]?[0-9]+)?)|(\?)|(\.)' 
    our_match = re.match(numb_re,numstring)
    if our_match:
        a,base_num,b,c,dad,dbd,esd,exp,q,dot = our_match.groups()
    #    print "Debug: %s -> %s" % (numstring, `our_match.groups()`)
    else:
        return None,None
    if dot or q: return None,None     #a dot or question mark
    if exp:          #has exponent 
       exp = string.replace(exp,"d","e")     # mop up old fashioned numbers
       exp = string.replace(exp,"D","e")
       base_num = base_num + exp
    #print "Debug: have %s for base_num from %s" % (base_num,numstring)
    base_num = float(base_num)
    # work out esd, if present.
    if esd:
        esd = float(esd[1:-1])    # no brackets
        if dad:                   # decimal point + digits
            esd = esd * (10 ** (-1* len(dad)))
        if exp:
            esd = esd * (10 ** (float(exp[1:])))
    return base_num,esd

def float_with_esd(inval):
    if isinstance(inval,basestring):
        j = inval.find("(")
        if j>=0:  return float(inval[:j])
    return float(inval)
        
    
                
def convert_type(definition):
    """Convert value to have the type given by definition"""
    #extract the actual required type information
    container = definition['_type.container']
    dimension = definition.get('_type.dimension',StarFile.StarList([]))
    structure = interpret_structure(definition['_type.contents'])
    if container == 'Single':   #a single value to convert
        return convert_single_value(structure)
    elif container == 'List':   #lots of the same value
        return convert_list_values(structure,dimension)
    elif container == 'Multiple': #no idea 
        return None
    elif container in ('Array','Matrix'): #numpy array
        return convert_matrix_values(structure)
    return lambda a:a    #unable to convert

def convert_single_value(type_spec):
    """Convert a single item according to type_spec"""
    if type_spec == 'Real':
        return float_with_esd
    if type_spec in ('Count','Integer','Index','Binary','Hexadecimal','Octal'):
        return int
    if type_spec == 'Complex':
        return complex
    if type_spec == 'Imag':
        return lambda a:complex(0,a)
    if type_spec in ('Code','Name','Tag'):  #case-insensitive -> lowercase
        return lambda a:a.lower()
    return lambda a:a   #can't do anything numeric
    
def convert_list_values(structure,dimension):
    """Convert the values according to the element
       structure given in [[structure]]"""
    if isinstance(structure,basestring):   #simple repetition
        func_def =  "element_convert = convert_single_value('%s')" % structure
    else:
        func_def =       "def element_convert(element):\n"
        func_def +=      "   final_val = []\n"   
        for pos_no in range(len(structure)):
            func_def +=  "   final_val.append("
            type_spec = structure[pos_no]
            if type_spec == 'Real':
                cf = "float_with_esd("
            elif type_spec in ('Count','Integer','Index','Binary','Hexadecimal','Octal'):
                cf = 'int('
            elif type_spec == 'Complex':
                cf = 'complex('
            elif type_spec == 'Imag':
                cf = 'complex(0,'
            elif type_spec in ('Code','Name','Tag'):
                cf = '('
            else: cf = ''
            func_def += cf
            func_def += "element[%d]" % pos_no
            if "(" in cf: func_def +=")"
            if type_spec in ('Code','Name','Tag'):
                func_def +=".lower()"
            func_def +=")\n"  # close append
        func_def +=      "   return final_val\n"
    print func_def
    exec func_def in globals() #(re)defines element_convert in global namespace
    if len(dimension)> 0 and int(dimension[0]) != 1:
        return lambda a: map(element_convert,a)
    else: return element_convert

def convert_matrix_values(valtype):
    """Convert a dREL String or Float valued List structure to a numpy matrix structure"""
    # first convert to numpy array, then let numpy do the work
    try: import numpy
    except:
        return lambda a:a   #can't do it
    func_def =     "def matrix_convert(a):\n"
    func_def +=    "    import numpy\n"
    func_def +=    "    p = numpy.array(a)\n"
    if valtype == 'Real':
        func_def+= "    return p.astype('float')\n"
    elif valtype == 'Integer':
        func_def +="    return p.astype('int')\n"
    elif valtype == 'Complex':
        func_def +="    return p.astype('complex')\n"
    else:
        raise ValueError, 'Unknown matrix value type'
    exec func_def  #matrix convert is defined
    return matrix_convert    
        
def interpret_structure(struc_spec):
    """Interpret a DDLm structure specification"""
    import TypeContentsParser as t
    p = t.TypeParser(t.TypeParserScanner(struc_spec))
    return getattr(p,"input")()
    

# A utility function to append to item values rather than replace them
def update_value(base_dict,new_items):
    for new_key in new_items.keys():
        if base_dict.has_key(new_key):
            base_dict[new_key].extend(new_items[new_key])
        else:
            base_dict[new_key] = new_items[new_key]

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

# listify strings - used surprisingly often
def listify(item):
    if isinstance(item,basestring): return [item]
    else: return item

# given a list of search items, return a list of items 
# actually contained in the given data block
def filter_present(namelist,datablocknames):
    return filter(lambda a:a in datablocknames,namelist)

# merge ddl dictionaries.  We should be passed filenames or CifFile
# objects
def merge_dic(diclist,mergemode="replace",ddlspec=None):
    dic_as_cif_list = []
    for dic in diclist:
        if not isinstance(dic,CifFile) and \
           not isinstance(dic,basestring):
               raise TypeError, "Require list of CifFile names/objects for dictionary merging"
        if not isinstance(dic,CifFile): dic_as_cif_list.append(CifFile(dic))
        else: dic_as_cif_list.append(dic)
    # we now merge left to right
    basedic = dic_as_cif_list[0]
    if basedic.has_key("on_this_dictionary"):   #DDL1 style only
        for dic in dic_as_cif_list[1:]:
           basedic.merge(dic,mode=mergemode,match_att=["_name"])
    elif len(basedic.keys()) == 1:                     #One block: DDL2 style
        old_block = basedic[basedic.keys()[0]]
        for dic in dic_as_cif_list[1:]:
           new_block = dic[dic.keys()[0]]
           basedic.merge(dic,mode=mergemode,
                         single_block=[basedic.keys()[0],dic.keys()[0]],
                         match_att=["_item.name"],match_function=find_parent)
    return CifDic(basedic)

def find_parent(ddl2_def):
    if not ddl2_def.has_key("_item.name"):
       return None 
    if isinstance(ddl2_def["_item.name"],basestring):
        return ddl2_def["_item.name"]
    if not ddl2_def.has_key("_item_linked.child_name"):
        raise CifError("Asked to find parent in block with no child_names")
    if not ddl2_def.has_key("_item_linked.parent_name"):
        raise CifError("Asked to find parent in block with no parent_names")
    result = filter(lambda a:a not in ddl2_def["_item_linked.child_name"],ddl2_def["_item.name"]) 
    if len(result)>1 or len(result)==0:
        raise CifError("Unable to find single unique parent data item")
    return result[0]


def ReadCif(filename,grammar='auto',scantype='standard',scoping='instance',standard='CIF'):
    """ Read in a CIF file, returning a `CifFile` object.  

    * `filename` may be a URL, a file
    path on the local system, or any object with a `read` method.

    * `grammar` chooses the CIF grammar variant. `1.0` is the original 1992 grammar and `1.1`
    is identical except for the exclusion of square brackets as the first characters in
    undelimited datanames. `2.0` will read files in the CIF2.0 standard, and `STAR2` will
    read files according to the STAR2 publication.  If grammar is `None`, autodetection
    will be attempted in the order `2.0`, `1.1` and `1.0`. This will always succeed for 
    properly-formed CIF2.0 files.  Note that only Unicode characters in the basic multilingual
    plane are recognised (this will be fixed when PyCIFRW is ported to Python 3).

    * `scantype` can be `standard` or `flex`.  `standard` provides pure Python parsing at the
    cost of a factor of 10 or so in speed.  `flex` will tokenise the input CIF file using
    fast C routines, but is not available for CIF2/STAR2 files.  Note that running PyCIFRW in 
    Jython uses native Java regular expressions
    to provide a speedup regardless of this argument (and does not yet support CIF2).

    * `scoping` is only relevant where nested save frames are expected (STAR2 only). 
    `instance` scoping makes nested save frames
    invisible outside their hierarchy, allowing duplicate save frame names in separate
    hierarchies. `dictionary` scoping makes all save frames within a data block visible to each 
    other, thereby restricting all save frames to have unique names.
    Currently the only recognised value for `standard` is `CIF`, which when set enforces a
    maximum length of 75 characters for datanames and has no other effect. """

    finalcif = CifFile(scoping=scoping,standard=standard)
    return StarFile.ReadStar(filename,prepared=finalcif,grammar=grammar,scantype=scantype)
    #return StarFile.StarFile(filename,maxlength,scantype=scantype,grammar=grammar,**kwargs)

class CifLoopBlock(StarFile.LoopBlock):
    def __init__(self,data=(),**kwargs):
        super(CifLoopBlock,self).__init__(data,**kwargs)

#No documentation flags

